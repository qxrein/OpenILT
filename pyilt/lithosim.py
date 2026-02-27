"""
Thin wrapper around the original `pylitho.simple.LithoSim` to provide an
Abbe-style interface for flare-aware ILT and, optionally, integration with
TorchLitho.

By default, the `abbe` method simply reuses the nominal printed image from
the underlying simulator as a proxy for the diffraction-limited image. For
true Abbe imaging, replace `abbe` with TorchLitho's Abbe model.
"""

import sys

sys.path.append(".")

import torch

from pylitho.simple import LithoSim as _BaseLithoSim


class LithoSim(_BaseLithoSim):
    def __init__(self, config="./config/lithosimple.txt"):
        super().__init__(config)

    def abbe(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Abbe imaging model placeholder.

        Parameters
        ----------
        mask : torch.Tensor
            Mask tensor [H, W] or [B, 1, H, W] (single-sample).

        Returns
        -------
        I_diff : torch.Tensor
            Diffraction-limited image [B, 1, H, W].

        Notes
        -----
        - Currently returns the nominal printed image from the underlying
          simulator as a proxy for I_diff.
        - For true Abbe imaging, replace this method with a call into
          TorchLitho's Abbe implementation.
        """
        # Ensure 2D mask for the base simulator
        if mask.dim() == 4:
            # Assume single-sample [B, 1, H, W]
            mask_2d = mask[0, 0]
        elif mask.dim() == 2:
            mask_2d = mask
        else:
            raise ValueError(f"Unsupported mask shape for abbe(): {mask.shape}")

        printed_nom, _, _ = super().forward(mask_2d)

        if printed_nom.dim() == 2:
            printed_nom = printed_nom.unsqueeze(0).unsqueeze(0)

        return printed_nom

