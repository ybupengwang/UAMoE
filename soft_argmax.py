import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftArgmax(nn.Module):
    def __init__(self, window_size=16, temperature=0.01,sigma_factor=0.25):
        """
        Soft-Argmax 计算类，基于局部加权平均的方法从热图中提取关键点坐标。

        参数：
        - window_size: 选取的局部窗口大小
        - temperature: Softmax 温度参数，越小对峰值位置越敏感
        """
        super().__init__()
        self.window_size = window_size
        self.temperature = temperature
        self.sigma_factor = sigma_factor

    def forward(self, heatmap):
        """
        计算 Soft-Argmax，返回归一化的关键点坐标 (x, y)。

        参数：
        - heatmap: (B, C, H, W) 形状的热图

        返回：
        - coords: (B, C, 2) 形状的张量，表示关键点坐标
        - spread: (B, C, 1) 形状的张量，表示局部方差（不确定性）
        """
        B, C, H, W = heatmap.shape

        # Step 1: 找到每个关键点的最大值索引
        heatmap_flatten = heatmap.view(B, C, -1)  # (B, C, H*W)
        max_indices = torch.argmax(heatmap_flatten, dim=-1)  # (B, C)
        max_x = max_indices % W
        max_y = max_indices // W

        # Step 2: 局部加权平均（soft-argmax in local window）
        xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        xx = xx.to(heatmap.device).float()
        yy = yy.to(heatmap.device).float()

        coords = torch.zeros((B, C, 2), device=heatmap.device)
        spread = torch.zeros((B, C, 1), device=heatmap.device)
        cov = torch.zeros((B, C, 2, 2), device=heatmap.device)
        for b in range(B):
            for c in range(C):
                x0 = max_x[b, c].item()
                y0 = max_y[b, c].item()

                # 局部窗口范围
                xmin = max(x0 - self.window_size // 2, 0)
                xmax = min(x0 + self.window_size // 2, W)
                ymin = max(y0 - self.window_size // 2, 0)
                ymax = min(y0 + self.window_size // 2, H)

                # 提取局部窗口
                patch = heatmap[b, c, ymin:ymax, xmin:xmax]
                patch = patch / self.temperature
                patch = F.softmax(patch.view(-1), dim=0).view_as(patch)

                # 对应坐标窗口
                x_patch = xx[ymin:ymax, xmin:xmax]
                y_patch = yy[ymin:ymax, xmin:xmax]



                # 坐标加权平均
                x_mean = (x_patch * patch).sum()
                y_mean = (y_patch * patch).sum()
                coords[b, c, 0] = x_mean / (W - 1)
                coords[b, c, 1] = y_mean / (H - 1)

                # 计算方差（不确定性）
                x_diff = (x_patch - x_mean) ** 2
                y_diff = (y_patch - y_mean) ** 2
                var = ((x_diff + y_diff) * patch).sum()
                spread[b, c, 0] = var
                #计算协方差
                x_diff = x_patch - x_mean  # (window_h, window_w)
                y_diff = y_patch - y_mean

                var_xx = (patch * x_diff * x_diff).sum()
                var_yy = (patch * y_diff * y_diff).sum()
                cov_xy = (patch * x_diff * y_diff).sum()

                cov[b, c, 0, 0] = var_xx
                cov[b, c, 1, 1] = var_yy
                cov[b, c, 0, 1] = cov_xy
                cov[b, c, 1, 0] = cov_xy

        return coords, cov,spread

