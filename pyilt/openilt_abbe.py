"""
OpenILT native kernel-based Abbe lithography for flare-aware ILT.

Uses pylitho.simple.LithoSim's precomputed kernels, which are derived from
a hard-edged circular pupil (Abbe formulation). Fully differentiable via
the kernel convolution's autograd backward. Use this for the "single most
impactful" experiment: flare-aware ILT with OpenILT's native Abbe pupil.
"""

import sys

sys.path.insert(0, ".")

import torch
import torch.nn as nn

from pycommon.settings import DEVICE, REALTYPE
from pylitho.simple import LithoSim as _BaseLithoSim, _LithoSim


class OpenILTAbbe(nn.Module):
    """
    OpenILT's native kernel-based Abbe imaging for I_diff.

    The kernels are precomputed from a hard-edged circular pupil (Abbe
    formulation). Returns the aerial image (before resist) so it can be
    used as I_diff in the flare model: I = I_diff + alpha * (PSF * I_diff).
    Fully differentiable.
    """

    def __init__(self, config="./config/lithosimple.txt"):
        super().__init__()
        self._litho = _BaseLithoSim(config)

    def abbe(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Return diffraction-limited aerial image from kernel-based Abbe model.

        Parameters
        ----------
        mask : torch.Tensor
            [H, W] or [B, 1, H, W]

        Returns
        -------
        I_diff : torch.Tensor
            Aerial image [1, 1, H, W], same spatial size as mask.
        """
        if mask.dim() == 4:
            mask_2d = mask[0, 0]
        elif mask.dim() == 2:
            mask_2d = mask
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

        # Ensure mask on same device as litho
        mask_2d = mask_2d.to(DEVICE).to(REALTYPE)

        # Get aerial image (before resist sigmoid) from kernel convolution
        aerial = _LithoSim.apply(
            mask_2d,
            self._litho._config["DoseNom"],
            self._litho._kernels["focus"].kernels,
            self._litho._kernels["focus"].scales,
            self._litho._config["KernelNum"],
            self._litho._kernels["combo CT focus"].kernels,
            self._litho._kernels["combo CT focus"].scales,
            1,
            self._litho._kernels["combo focus"].kernels,
            self._litho._kernels["combo focus"].scales,
            1,
        )

        if aerial.dim() == 2:
            aerial = aerial.unsqueeze(0).unsqueeze(0)
        return aerial.to(mask.device)

    def forward(self, mask: torch.Tensor):
        """Compatible with evaluation: (nom, max, min)."""
        I = self.abbe(mask)
        return I, I, I
