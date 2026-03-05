import torch
import torch.nn as nn


class SRSNetLite(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout=0.05):
        super().__init__()

        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2,
            groups=d_model
        )

        self.gate = nn.Conv1d(d_model, d_model, 1, groups=d_model)

        # learnable residual strength
        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B*N, P, D]
        res = x

        x = x.transpose(1, 2)  # [B*N, D, P]

        y = self.conv(x)

        # soft modulation (won't kill smooth datasets)
        g = 1 + 0.3 * torch.tanh(self.gate(x))

        y = y * g + self.alpha * x
        y = self.dropout(y)

        y = y.transpose(1, 2)  # [B*N, P, D]
        y = self.norm(y + res)
        return y