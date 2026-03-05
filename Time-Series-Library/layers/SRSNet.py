import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvg(nn.Module):
    """
    Moving average for trend extraction
    """
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        # x: [B, L, C]
        x = x.permute(0, 2, 1)      # [B, C, L]
        trend = self.avg(x)
        trend = trend.permute(0, 2, 1)  # [B, L, C]
        return trend


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return trend, seasonal


class SRSNet(nn.Module):
    """
    Plug-and-play SRSNet for time series forecasting
    """
    def __init__(
        self,
        d_model,
        kernel_size=25,
        dropout=0.1,
        use_trend=True
    ):
        super().__init__()
        self.use_trend = use_trend
        self.decomp = SeriesDecomp(kernel_size)

        self.season_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.resid_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        if use_trend:
            self.trend_net = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, L, C]
        """
        trend, seasonal = self.decomp(x)

        s_out = self.season_net(seasonal)
        r_out = self.resid_net(x - seasonal)

        if self.use_trend:
            t_out = self.trend_net(trend)
            out = s_out + r_out + t_out
        else:
            out = s_out + r_out

        out = self.layer_norm(out + x)  # residual
        return out