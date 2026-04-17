import numpy as np
import torch
import random
def get_markposion_fromtxt(point_num, path):
    flag = 0
    x_pos = []
    y_pos = []
    with open(path) as note:
        for line in note:
            if flag >= point_num:
                break
            else:
                flag += 1
                x, y = [float(i) for i in line.split(',')]
                x_pos.append(x)
                y_pos.append(y)
        x_pos = np.array(x_pos)
        y_pos = np.array(y_pos)
    return x_pos, y_pos


def get_prepoint_from_htmp(heatmaps, scal_ratio_w, scal_ratio_h,resize_w,resize_h):
    pred = np.zeros((19, 2))
    for i in range(19):
        heatmap = heatmaps[i]
        pre_y, pre_x = np.where(heatmap == np.max(heatmap))
        pred[i][1] = pre_y[0] * (scal_ratio_h*resize_h-1)/(resize_h-1)
        pred[i][0] = pre_x[0] * (scal_ratio_w*resize_w-1)/(resize_w-1)
    return pred

def get_prepoint_from_htmp_topk(heatmaps, scal_ratio_w, scal_ratio_h,topk = 10):
    num_kp, H, W = heatmaps.shape
    pred = np.zeros((19, 2))
    for i in range(19):
        heatmap = heatmaps[i]
        flat = heatmap.flatten()
        topk_indices = np.argpartition(-flat, topk)[:topk]
        y_coords = topk_indices // W
        x_coords = topk_indices % W
        mean_x = np.mean(x_coords) * scal_ratio_w
        mean_y = np.mean(y_coords) * scal_ratio_h
        pred[i] = [mean_x, mean_y]
    return pred

def get_prepoint_from_htmp_sampling(heatmaps, scal_ratio_w, scal_ratio_h, num_samples=10):
    """
    使用 Sampling-Argmax 方法从热图中获取坐标点

    Args:
        heatmaps (numpy.ndarray): (19, H, W) 的热图
        scal_ratio_w (float): 宽度缩放比例
        scal_ratio_h (float): 高度缩放比例
        num_samples (int): 采样次数（可以取 1~5 之间）

    Returns:
        pred (numpy.ndarray): (19, 2) 预测关键点坐标
    """
    num_keypoints, H, W = heatmaps.shape
    pred = np.zeros((num_keypoints, 2))

    for i in range(num_keypoints):
        heatmap = torch.tensor(heatmaps[i], dtype=torch.float32).view(-1)  # 展平 (H*W)
        prob = torch.softmax(heatmap, dim=0)  # 计算 softmax 作为概率分布

        # 从概率分布中采样 num_samples 个索引
        sampled_indices = torch.multinomial(prob, num_samples, replacement=True)

        # 计算采样坐标的均值
        y_samples = sampled_indices // W  # 计算 y 坐标
        x_samples = sampled_indices % W   # 计算 x 坐标

        pre_x = x_samples.float().mean().item() * scal_ratio_w
        pre_y = y_samples.float().mean().item() * scal_ratio_h

        pred[i] = [pre_x, pre_y]  # 存储预测坐标

    return pred

def convert_to_true_coords(x_pos, y_pos):
    """
    将 x_pos 和 y_pos 转换为符合 (batch_size, num_landmarks, 2) 形状的 true_coords。
    """
    assert len(x_pos) == len(y_pos), "x_pos 和 y_pos 长度不匹配"

    # (num_landmarks, 2) 形状
    true_coords = np.stack((x_pos, y_pos), axis=-1)

    # 转换成 PyTorch Tensor
    true_coords = torch.tensor(true_coords, dtype=torch.float32)

    return true_coords


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True