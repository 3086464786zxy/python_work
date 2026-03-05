import torch
import torch.nn as nn


class SpectralReweight(nn.Module):
    """
    Patch token spectral reweighting
    Stable improvement for PatchTST
    """

    def __init__(self, d_model):
        super().__init__()

        # learnable frequency scaling
        self.scale = nn.Parameter(torch.ones(1, 1, d_model))

        # residual strength
        self.alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        """
        x: [B, N_patch, D]
        """

        # FFT along patch dimension
        freq = torch.fft.rfft(x, dim=1)

        # energy
        energy = torch.abs(freq)

        # normalize energy
        energy_mean = energy.mean(dim=1, keepdim=True) + 1e-6
        norm_energy = energy / energy_mean

        # spectral weighting
        freq = freq * (1 + self.scale * norm_energy)

        # inverse FFT
        x_spec = torch.fft.irfft(freq, n=x.size(1), dim=1)

        # residual mixing (VERY important)
        return x + self.alpha * x_spec