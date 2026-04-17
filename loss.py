import torch
import torch.nn as nn
from config import Config

class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight=False, topk=15, num_joints=Config.point_num, ema_momentum=0.9):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.num_joints = num_joints
        self.ema_momentum = ema_momentum

        self.register_buffer("running_err", torch.zeros(num_joints))

    def forward(self, output, target, target_weight=1):
        batch_size = output.size(0)
        num_joints = output.size(1)

        # output, target: (B, C, H, W)
        # 拆 C 维，得到列表，每个元素是 [B, H, W]
        heatmaps_pred = torch.unbind(output, dim=1)  # len=19
        heatmaps_gt = torch.unbind(target, dim=1)    # len=19

        joint_losses = []

        for idx in range(num_joints):
            pred = heatmaps_pred[idx]  # [B, 800, 640]
            gt = heatmaps_gt[idx]      # [B, 800, 640]

            if self.use_target_weight:
                weight = target_weight[:, idx].view(batch_size, 1, 1)  # broadcast
                joint_loss = 0.5 * self.criterion(pred * weight, gt * weight)
            else:
                joint_loss = self.criterion(pred, gt)
                joint_loss = joint_loss

            joint_loss = joint_loss.mean(dim=(1, 2))  # → [B]
            joint_losses.append(joint_loss.unsqueeze(1))  # → [B, 1]

            # 更新历史误差
            with torch.no_grad():
                cur_mean = joint_loss.mean()
                self.running_err[idx] = (self.ema_momentum * self.running_err[idx] + (1 - self.ema_momentum) * cur_mean)

        # 拼接所有关节点误差 → [B, num_joints]
        loss = torch.cat(joint_losses, dim=1)

        # 选出历史上 topk 难学点
        _, hard_joint_idx = torch.topk(self.running_err, k=self.topk, largest=True)
        if self.training :
            with open("hard_topk.txt", 'a') as f:
                # 写入表头
                f.write('\t'.join([str(i.item()) for i in hard_joint_idx]) + '\n')

        loss_hard = loss[:, hard_joint_idx]  # [B, topk]
        if self.topk ==19:
            return loss.mean()
        else:
            return loss_hard.mean()
class JointsOHKMCoorLoss(nn.Module):
    def __init__(self, use_target_weight=True, topk=5):
        super(JointsOHKMCoorLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.criterion = nn.MSELoss(reduction='none')

    def ohkm(self, loss):
        # loss: [batch_size, num_joints]
        ohkm_loss = 0.
        for i in range(loss.size(0)):
            sample_loss = loss[i]  # [num_joints]
            topk_val, topk_idx = torch.topk(sample_loss, k=self.topk, dim=0, sorted=False)
            topk_loss = torch.gather(sample_loss, 0, topk_idx)
            ohkm_loss += torch.sum(topk_loss) / self.topk
        ohkm_loss /= loss.size(0)
        return ohkm_loss

    def forward(self, output, target, target_weight=None):
        # output & target: [batch_size, num_joints, 2]
        loss = self.criterion(output, target)  # shape: [B, J, 2]
        loss = torch.sum(loss, dim=2)  # 对x和y求和，shape: [B, J]

        if self.use_target_weight and target_weight is not None:
            # target_weight: [B, J, 1] 或 [B, J]
            if target_weight.dim() == 3:
                target_weight = target_weight.squeeze(-1)
            loss = loss * target_weight

        return self.ohkm(loss)

