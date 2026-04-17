
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplingArgmax(nn.Module):
    def __init__(self, num_samples=10, init_temp=0.1, min_temp=0.01, decay=0.98, learnable_temp=False):
        """
        Sampling-Argmax 模块
        :param num_samples: 每次采样的数量
        :param init_temp: 初始温度参数
        :param min_temp: 最小温度
        :param decay: 温度衰减率
        :param learnable_temp: 是否使用可学习温度
        """
        super().__init__()
        self.num_samples = num_samples
        self.init_temp = init_temp
        self.min_temp = min_temp
        self.decay = decay
        self.learnable_temp = learnable_temp

        if learnable_temp:
            self.log_temperature = nn.Parameter(torch.tensor(0.0))  # 学习 log(T)
        else:
            self.current_temp = init_temp

    def update_temperature(self, epoch):
        """ 训练过程中动态降低温度 """
        if not self.learnable_temp:
            self.current_temp = max(self.min_temp, self.init_temp * (self.decay ** epoch))

    def forward(self, heatmap):
        """
        :param heatmap: 形状为 (B, C, H, W) 的热图
        :return: 预测的坐标 (B, C, 2) 归一化到 [0,1]
        """
        B, C, H, W = heatmap.shape

        # 计算 softmax 以获得概率分布
        temp = torch.exp(self.log_temperature) if self.learnable_temp else self.current_temp
        prob = F.softmax(heatmap.view(B, C, -1) / temp, dim=-1)  # (B, C, H*W)
        prob = prob.view(B * C, H*W)
        # 从概率分布中采样 num_samples 个点
        sampled_indices = torch.multinomial(prob, self.num_samples, replacement=True)  # (B, C, num_samples)
        sampled_indices = sampled_indices.view(B, C, self.num_samples)
        # 计算坐标
        y_idx = (sampled_indices // W).float() / H  # 归一化 y 坐标
        x_idx = (sampled_indices % W).float() / W  # 归一化 x 坐标

        # 取均值作为最终预测坐标
        pred_x = x_idx.mean(dim=-1)  # (B, C)
        pred_y = y_idx.mean(dim=-1)  # (B, C)

        return torch.stack([pred_x, pred_y], dim=-1)  # (B, C, 2)