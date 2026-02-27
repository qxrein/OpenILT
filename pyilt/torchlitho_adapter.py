import importlib
import os
import sys
from typing import Optional

import torch
import torch.nn as nn


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
        device: Optional[torch.device]
        if isinstance(mask, torch.Tensor):
            device = mask.device
        else:
            device = None

        ali = self.im.CalculateAerialImage()
        I = ali.Intensity  

        if device is not None:
            I = I.to(device)

        if I.dim() == 2:
            I = I.unsqueeze(0).unsqueeze(0)
        elif I.dim() == 3:
            I = I.unsqueeze(0)

        return I

