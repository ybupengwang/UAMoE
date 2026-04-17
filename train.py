import torch
import time
from config import Config
Config = Config()
from test import get_errors
from utils import convert_to_true_coords
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()


def rle_loss_residual(pred, target, coarse_coord, scale_factor=0.001):
    """
    pred: (B, 19, 4) -> [:2] 是 Δmean， [2:] 是 σ
    target: (B, 19, 2) -> GT坐标
    coarse_coord: (B, 19, 2)
    """
    delta_mean = pred[:, :, :2]
    sigma = pred[:, :, 2:]  # 必须保证预测值通过 softplus/sqrt 等方式保证 > 0

    # 复原 refined 坐标
    mean = coarse_coord.detach() + delta_mean * scale_factor

    # log(sigma + 1) 以避免负值损失（比 log(sigma) 更稳定）
    log_term = torch.log(sigma + 1.0)
    squared_term = (target - mean) ** 2 / sigma

    loss = 0.5 * (log_term + squared_term)
    return loss.mean()

def heatmap_dice_loss_topk(pred, target, topk=15, epsilon=1e-5,ema_momentum=0.9,selected_idxs=[3, 11, 18],guding = False):
    """
    pred:  shape (N, C, H, W)
    target: shape (N, C, H, W)
    topk: 对每个样本取损失最大的 topk 个关键点计算平均损失
    """
    if guding:
        # 只选取指定关键点通道
        pred = pred[:, selected_idxs, :, :]
        target = target[:, selected_idxs, :, :]
    # flatten to (N, C, H*W)
    pred = pred.view(pred.shape[0], pred.shape[1], -1)
    target = target.view(target.shape[0], target.shape[1], -1)
    # calculate per-keypoint dice loss (N, C)
    intersection = 2 * (pred * target).sum(dim=2)
    union = (pred ** 2).sum(dim=2) + (target ** 2).sum(dim=2)

    dice = (intersection + epsilon) / (union + epsilon)
    loss = 1 - dice  # (N, C)

    # 对每个样本，选择 topk 个最大 loss（困难点）
    if topk >= loss.size(1):  # 如果 topk 超出点数，则退化为普通平均
        return loss.mean()
    topk_loss,hard_joint_idx = torch.topk(loss, topk, dim=1)  # shape (N, topk)

    with open("hard_topkdice.txt", 'a') as f:
        f.write('\t'.join(map(str, hard_joint_idx.tolist())) + '\n')
    if guding:
        return 0.5 * topk_loss.mean() + 0.5 * loss.mean()
    if topk !=19:
        return topk_loss.mean()
    else:
        return  loss.mean()
def heatmap_dice_loss(pred, target, epsilon=1e-5):
    """
    pred:  shape (N, C, H, W)
    target: shape (N, C, H, W)
    """
    # flatten to (N, C, H*W)
    pred = pred.view(pred.shape[0], pred.shape[1], -1)
    target = target.view(target.shape[0], target.shape[1], -1)

    # calculate intersection and union
    intersection = 2 * (pred * target).sum(dim=2)
    union = (pred ** 2).sum(dim=2) + (target ** 2).sum(dim=2)

    dice = (intersection + epsilon) / (union + epsilon)
    loss = 1 - dice  # (N, C)

    return loss.mean()                    #如果单独训练的某个点的时候可以在这里进行修改




def train_model(model,soft_argmax, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs,epoch, trans = Config.trans, struct_biaozhi = Config.struct_biaozhi):
    print('Epoch{}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    model.train()
    loss_heat_total = 0
    loss_coo_total =0
    loss_coo_tran_total = 0
    loss_kl_total = 0
    dice_toal = 0
    loss_total = 0
    start_time = time.time()
    for i, (img, img_w,img_h,heatmaps, heatmaps_refine,  img_name, x_all, y_all, gt_x,gt_y,heatmaps_hrnet) in enumerate(train_loader):
        img = img.cuda(Config.GPU)
        img_h = img_h.cuda(Config.GPU)
        img_w = img_w.cuda(Config.GPU)
        heatmaps = heatmaps.cuda(Config.GPU)
        heatmaps_refine = heatmaps_refine.cuda(Config.GPU)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs, outputs_refine,output_struct,refine_hp_2,refined_coords,log_sigma,feature_tran32,kl_loss,delta,coarse_coord= model(img)
            loss_heat = criterion(outputs, heatmaps)   #热图损失
            loss_heat_dice = heatmap_dice_loss_topk(outputs, heatmaps)  # 热图dice损失

            batch_size, num_landmarks, height, width = heatmaps.shape
            true_coords = convert_to_true_coords(gt_x,gt_y).cuda(Config.GPU)
            true_coords[:, :, 0] /= (img_w[:,None]-1)  #
            true_coords[:, :, 1] /= (img_h[:,None]-1)  #归一化坐标
            #结构特征损失
            if struct_biaozhi:

                with torch.no_grad():
                    delta_gt = true_coords[:, :, None, :] - true_coords[:, None, :, :]  # (B, N, N, 2)
                    delta_gt_feature = delta_gt.view(delta_gt.size(0), -1)  # (B, N*N*2)
                struct_loss = F.mse_loss(output_struct,delta_gt_feature)
            else:
                struct_loss = torch.tensor(0.0)
            #专家开启 微调坐标
            if trans:
                #loss_coo_tran = F.mse_loss(refined_coords,true_coords)
                loss_coo_tran = rle_loss_residual(delta, true_coords,coarse_coord)
            else:
                loss_coo_tran =kl_loss= torch.tensor(0.0)
            loss =  loss_heat +  loss_heat_dice + 0*struct_loss + 0.0*loss_coo_tran +  0.0*kl_loss #注意test坐标

            #各类损失和
            loss_total += loss.item()
            loss_heat_total +=loss_heat.item() #第一热图mse损失
            loss_coo_total += loss_coo_tran.item() #专家微调损失
            loss_coo_tran_total += kl_loss.item() #专家平衡损失
            loss_kl_total +=struct_loss.item()  #结构形状一致性损失
            dice_toal += loss_heat_dice.item()  #第一热图dice损失
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # 梯度裁剪（防止爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    loss_avg = loss_total / len(train_loader)
    loss_heat_avg = loss_heat_total/len(train_loader)
    loss_coo_avg = loss_coo_total/len(train_loader)
    loss_coo_tran_avg = loss_coo_tran_total/len(train_loader)
    loss_kl_avg = loss_kl_total/len(train_loader)
    dice_avg = dice_toal/len(train_loader)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'训练 Loss: {loss_avg:.6f} 用时：{elapsed_time:.4f}')
    get_errors(model, soft_argmax,test_loader, Config.test_gt_dir1, Config.save_results_path)
    scheduler.step()
