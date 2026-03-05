import torch
import torch.nn as nn
import torch.nn.functional as F


class SRSNetLite(nn.Module):
    """
    SRSNet-Lite v2_4
    - 多尺度 depthwise conv
    - tanh 门控
    - 残差缩放 (0.5)
    - 输出 LayerNorm
    """

    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.k = kernel_size

        # 多尺度卷积核
        kernels = [self.k, self.k * 2 + 1]  # 保持与你 v2_3 一致
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kk,
                padding=kk // 2,
                groups=d_model,   # depthwise
                bias=False
            )
            for kk in kernels
        ])

        # 融合多尺度特征
        self.fuse = nn.Conv1d(
            in_channels=d_model * len(kernels),
            out_channels=d_model,
            kernel_size=1,
            bias=False
        )

        # 门控
        self.gate_fc = nn.Linear(d_model, d_model)

        # 归一化（新增）
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        B, L, D = x.shape

        # [B, D, L]
        x_t = x.transpose(1, 2)

        feats = []
        for conv in self.convs:
            feats.append(conv(x_t))

        # [B, D * K, L]
        feats = torch.cat(feats, dim=1)

        # 融合回 [B, D, L]
        srs_feat = self.fuse(feats)

        # 回到 [B, L, D]
        srs_feat = srs_feat.transpose(1, 2)

        # LayerNorm（新增）
        srs_feat = self.norm(srs_feat)

        # tanh 门控（替换 sigmoid）
        gate = torch.tanh(self.gate_fc(x))

        # 残差缩放（新增 0.5）
        out = x + 0.5 * gate * srs_feat

        return out