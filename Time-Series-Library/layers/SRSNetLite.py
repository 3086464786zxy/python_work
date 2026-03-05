import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        # x: [B, L, C]
        x = x.permute(0, 2, 1)     # [B, C, L]
        trend = self.avg(x)
        trend = trend.permute(0, 2, 1)
        return trend


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return trend, seasonal


class SRSNetLite(nn.Module):
    """
    SRSNet-Lite: short-sequence optimized
    input:  [B, L, D]
    output: [B, L, D]
    """
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()

        self.decomp = SeriesDecomp(kernel_size)

        # 轻量季节分支
        self.season_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 轻量残差分支
        self.resid_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 门控融合（核心）
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        trend, seasonal = self.decomp(x)

        s_out = self.season_net(seasonal)
        r_out = self.resid_net(x - seasonal)

        gate = self.gate(torch.cat([s_out, r_out], dim=-1))
        out = gate * s_out + (1 - gate) * r_out

        out = self.norm(out + x)   # 稳定残差
        return out
