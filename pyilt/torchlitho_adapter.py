import importlib
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# Make sure TorchLitho's src directory is on the path
_this_dir = os.path.dirname(__file__)
_torchlitho_root = os.path.join(os.path.dirname(_this_dir), "TorchLitho", "src")
if _torchlitho_root not in sys.path:
    sys.path.append(_torchlitho_root)

# Ensure TorchLitho's `models` package is importable
_torchlitho_models = os.path.join(_torchlitho_root, "models")
if _torchlitho_models not in sys.path:
    sys.path.append(_torchlitho_models)

# Provide shims for TorchLitho modules that are imported by bare name
# inside the litho package (e.g. `from Source import Source`).
try:
    import Source as _tl_source  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - shim path
    _tl_source = importlib.import_module("models.litho.Source")
    sys.modules["Source"] = _tl_source

try:
    import ProjectionObjective as _tl_po  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - shim path
    _tl_po = importlib.import_module("models.litho.ProjectionObjective")
    sys.modules["ProjectionObjective"] = _tl_po

from models.litho.ImagingModel import ImagingModel  # type: ignore


class TorchLithoAbbe(nn.Module):
    """
    Thin adapter that exposes TorchLitho's Abbe imaging as an `abbe(mask)` API
    compatible with `FlareAwareILT`.

    Notes
    -----
    - For now this uses TorchLitho's internal mask configuration and ignores
      the incoming `mask` tensor contents (it treats `mask` only as a shape /
      device hint). This is sufficient to demonstrate explicit TorchLitho
      Abbe/Hopkins integration for the flare-aware framework.
    - To drive TorchLitho directly from the OpenILT mask field, you would need
      to map the mask tensor into TorchLitho's `Mask` representation (e.g.,
      via its pixel or spectrum interfaces).
    """

    def __init__(self, image_method: str = "abbe"):
        super().__init__()
        self.im = ImagingModel()
        # Force Abbe or Hopkins imaging in TorchLitho
        self.im.Numerics.ImageCalculationMethod = image_method

    def abbe(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Abbe/Hopkins imaging entry point used by `FlareAwareILT`.

        Parameters
        ----------
        mask : torch.Tensor
            Mask tensor [B,1,H,W] or [H,W]. Only its device is used here.

        Returns
        -------
        I_diff : torch.Tensor
            Diffraction-limited aerial image [B,1,H,W] on the same device.
        """
        """
        True TorchLitho Abbe imaging driven by the input mask tensor.

        This uses TorchLitho's `MaskType='2DPixel'` path, which expects an odd
        sampling size. If the incoming mask has even spatial dimensions, we
        pad by 1 pixel on the right/bottom, run TorchLitho, then crop back.
        """
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor")

        # Accept [H,W] or [B,1,H,W] (single batch)
        if mask.dim() == 4:
            if mask.size(0) != 1 or mask.size(1) != 1:
                raise ValueError(f"Expected mask shape [1,1,H,W], got {mask.shape}")
            m2d = mask[0, 0]
        elif mask.dim() == 2:
            m2d = mask
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

        H, W = int(m2d.shape[-2]), int(m2d.shape[-1])
        pad_h = 1 if (H % 2 == 0) else 0
        pad_w = 1 if (W % 2 == 0) else 0
        if pad_h or pad_w:
            # Pad in 4D for replicate mode, then squeeze back
            m4d = m2d.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            m4d = F.pad(m4d, (0, pad_w, 0, pad_h), mode="replicate")
            m2d_pad = m4d[0, 0]
        else:
            m2d_pad = m2d

        Hp, Wp = int(m2d_pad.shape[-2]), int(m2d_pad.shape[-1])

        # Configure TorchLitho to use the pixel mask spectrum path
        self.im.Mask.MaskType = "2DPixel"
        self.im.Mask.Nf = Wp
        self.im.Mask.Ng = Hp
        # Feature is the complex transmittance map
        self.im.Mask.Feature = m2d_pad.to(torch.complex64)

        # Make mask/wafer sampling consistent with the tensor size
        self.im.Numerics.SampleNumber_Mask_X = Wp
        self.im.Numerics.SampleNumber_Mask_Y = Hp
        self.im.Numerics.SampleNumber_Wafer_X = Wp
        self.im.Numerics.SampleNumber_Wafer_Y = Hp

        ali = self.im.CalculateAerialImage()
        I = ali.Intensity

        # TorchLitho returns [Z, X, Y] after transpose in Calculate2DAerialImage
        if I.dim() == 3:
            I = I[0]  # nominal focus slice
        if I.dim() != 2:
            raise ValueError(f"Unexpected TorchLitho intensity shape: {I.shape}")

        # Crop back to original size if padded
        if pad_h or pad_w:
            I = I[:H, :W]

        return I.unsqueeze(0).unsqueeze(0)

    def forward(self, mask: torch.Tensor):
        """
        Simple forward compatible with OpenILT evaluation utilities.

        Returns nominal, max, and min images identical to the Abbe aerial
        image so that L2 / PV metrics can be computed without a full resist
        / process-window model.
        """
        I = self.abbe(mask)
        return I, I, I

