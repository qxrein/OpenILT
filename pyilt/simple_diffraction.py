"""
Differentiable diffraction proxy for flare-aware ILT when full lithography
is not needed or is not differentiable (e.g., TorchLitho on CPU).

Uses a separable Gaussian blur (two 1D passes) for faster compute. Fully
differentiable; gradients flow through the fidelity term to the mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_1d(sigma: float, size: int) -> torch.Tensor:
    """Create normalized 1D Gaussian kernel."""
    if size % 2 == 0:
        size += 1
    half = size // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return (k / k.sum()).reshape(1, 1, -1)


class SimpleDiffractionLitho(nn.Module):
    """
    Differentiable lithography proxy: I_diff = separable Gaussian blur of mask.

    Uses two 1D convolutions instead of one 2D for faster compute (~O(2k) vs O(k²)).
    Kernels pre-computed at init; gradients flow to mask.
    """

    def __init__(self, sigma: float = 2.0, kernel_size: int = 15):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        k1d = _gaussian_1d(sigma, kernel_size)  # [1,1,W]
        self.register_buffer("_kh", k1d.reshape(1, 1, 1, -1))  # horizontal
        self.register_buffer("_kv", k1d.reshape(1, 1, -1, 1))  # vertical

    def abbe(self, mask: torch.Tensor) -> torch.Tensor:
        """Return diffraction-limited image as separable Gaussian blur of mask."""
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)

        kh = self._kh.to(mask.device, mask.dtype)
        kv = self._kv.to(mask.device, mask.dtype)
        pad_w = kh.shape[-1] // 2
        pad_h = kv.shape[-2] // 2
        t = F.conv2d(mask, kh, padding=(0, pad_w))
        I_diff = F.conv2d(t, kv, padding=(pad_h, 0))
        return I_diff

    def forward(self, mask: torch.Tensor):
        """Compatible with evaluation utilities: (nom, max, min)."""
        I = self.abbe(mask)
        return I, I, I
