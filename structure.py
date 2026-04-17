import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal


class StructureEncoder(nn.Module):
    def __init__(self, num_points=19):
        super().__init__()
        input_dim = num_points * num_points*2  # delta展开后维度

        hidden_dim = 256
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, delta_feat):
        # delta_feat: (B, N*N)

        correction = self.net(delta_feat)  # (B, N*N)
        return correction



