"""
Differentiable diffraction proxy for flare-aware ILT when full lithography
is not needed or is not differentiable (e.g., TorchLitho on CPU).

I_diff = mask (identity) provides full gradient flow through the fidelity
term, so the optimizer can actually update the mask. Use this for:
- Proof-of-concept runs where you need visible optimizer progress
- Debugging gradient flow before integrating full Abbe/Hopkins
"""

import torch
import torch.nn as nn


class SimpleDiffractionLitho(nn.Module):
    """
    Differentiable lithography proxy: I_diff = mask.

    Gradients flow from loss through printed → I_total → I_diff → mask.
    Use this instead of TorchLithoAbbe when the full imaging model is
    non-differentiable or too slow, and you need the optimizer to update
    the mask.
    """

    def __init__(self):
        super().__init__()

    def abbe(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Return mask as I_diff (identity, fully differentiable).

        Parameters
        ----------
        mask : torch.Tensor
            [H,W] or [B,1,H,W]

        Returns
        -------
        I_diff : torch.Tensor
            Same as mask, [1,1,H,W] for compatibility.
        """
        if mask.dim() == 2:
            return mask.unsqueeze(0).unsqueeze(0)
        if mask.dim() == 4:
            return mask
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, mask: torch.Tensor):
        """Compatible with evaluation utilities: (nom, max, min)."""
        I = self.abbe(mask)
        return I, I, I
