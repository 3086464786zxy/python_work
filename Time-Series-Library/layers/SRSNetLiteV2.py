import torch
import torch.nn as nn


class SRSNetLite(nn.Module):
    """
    SRSNet-Lite v2.1 (Short-Sequence Optimized)
    - 单尺度深度卷积
    - 轻量通道门控（无 MLP）
    - 残差稳定
    - 极低参数量，防过拟合
    输入: [B*N, P, D]
    输出: [B*N, P, D]
    """

    def __init__(self, d_model, kernel_size=3, dropout=0.05):
        super().__init__()

        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2,
            groups=d_model
        )

        # 极简 gate：1x1 depthwise + sigmoid
        self.gate = nn.Conv1d(d_model, d_model, 1, groups=d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B*N, P, D]
        res = x

        x = x.transpose(1, 2)  # [B*N, D, P]
        y = self.conv(x)
        g = torch.sigmoid(self.gate(x))
        y = y * g + 0.3 * x
        #y = y * g + 1 * x
        y = self.dropout(y)
        y = y.transpose(1, 2)  # [B*N, P, D]

        y = self.norm(y + res)
        return y