import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from pyilt.flare_psf import FlarePSF


class FlareAwareILT(nn.Module):
    """
    Flare-aware inverse lithography solver with sensitivity-based adaptive
    weighting and flare regularization.

    Notes
    -----
    - This class follows the mathematical formulation provided in the prompt.
    - It does NOT rely on `SimpleILT.__init__`'s tiling/filter logic; its own
      configuration dictionary `cfg` is used instead.
    - The lithography simulator must be differentiable and should ideally
      provide an Abbe/Hopkins-style forward model via `abbe(mask)`. When that
      is not available, the nominal output of `litho(mask)` is used.
    """

    def __init__(self, cfg, litho, flare_psf: FlarePSF):
        """
        Parameters
        ----------
        cfg : dict
            Configuration dictionary with keys:
            - 'max_iters' : int
            - 'lr' : float
            - 'flare_weight_beta' : float
            - 'flare_reg_weight' : float
            - 'lambda1' : float
            - 'theta_M' : float
        litho : nn.Module
            Lithography simulator. Must be callable on a mask tensor and
            optionally provide an `.abbe(mask)` method.
        flare_psf : FlarePSF
            FlarePSF instance for flare intensity computation and PSF_SC kernel.
        """
        super().__init__()

        self.cfg = dict(cfg)
        self.litho = litho
        self.flare_psf = flare_psf

        self.beta = self.cfg.get("flare_weight_beta", 5.0)
        self.mu = self.cfg.get("flare_reg_weight", 0.1)
        self.lambda1 = self.cfg.get("lambda1", 1.0)
        self.lr = self.cfg.get("lr", 0.01)
        self.max_iters = int(self.cfg.get("max_iters", 800))
        self.theta_M = self.cfg.get("theta_M", 10.0)

        # Pre-compute PSF_SC kernel for flare regularization
        psf_sc = self.flare_psf.generate_shiraishi_psf()
        self.psf_sc = psf_sc.to(next(self.litho.parameters(), torch.zeros(1)).device)

    def compute_pvband(self, mask):
        """
        Placeholder PV band loss used in the total flare-aware loss.

        A full implementation should evaluate process corners (dose/focus)
        and compute the band between printedMax and printedMin contours.
        """
        if isinstance(mask, torch.Tensor):
            return torch.tensor(0.0, device=mask.device)
        return torch.tensor(0.0)

    def compute_sensitivity_map(self, mask, I_diff, I_flare):
        """
        Compute the flare sensitivity map

            S(r) = ||∇_M I_diff|| / (||∇_M I_diff|| + ||∇_M I_flare|| + ε)

        Parameters
        ----------
        mask : torch.Tensor
            Mask tensor [B, 1, H, W] with requires_grad enabled.
        I_diff : torch.Tensor
            Diffraction-limited nominal image [B, 1, H, W].
        I_flare : torch.Tensor
            Flare contribution [B, 1, H, W].
        """
        eps = 1e-8

        # If the imaging pipeline is not differentiably connected to the mask
        # (e.g. TorchLitho adapter returns a detached aerial image), fall back
        # to a uniform sensitivity map.
        if not (isinstance(I_diff, torch.Tensor) and I_diff.requires_grad):
            return torch.ones_like(mask)

        # Use the same mask tensor that was used to compute I_diff / I_flare.
        grad_I_diff = torch.autograd.grad(
            I_diff.sum(),
            mask,
            create_graph=False,
            retain_graph=True,
            allow_unused=True,
        )[0]

        grad_I_flare = torch.autograd.grad(
            I_flare.sum(),
            mask,
            create_graph=False,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if grad_I_diff is None:
            grad_I_diff = torch.zeros_like(mask)
        if grad_I_flare is None:
            grad_I_flare = torch.zeros_like(mask)

        S = torch.abs(grad_I_diff) / (torch.abs(grad_I_diff) + torch.abs(grad_I_flare) + eps)
        return S.detach()

    def compute_adaptive_weights(self, S, alpha):
        """
        Compute the adaptive weight map

            w(r) = S(r) / (1 + β·α(r))
        """
        return S / (1.0 + self.beta * alpha)

    def compute_flare_regularization(self, I_diff):
        """
        Flare regularization term

            L_flare = ∫ ||∇[PSF_SC * I_diff]||² dr
        """
        # PSF_SC * I_diff (ensure kernel is on the same device)
        psf_sc = self.psf_sc.to(I_diff.device)
        flare_contrib = F.conv2d(I_diff, psf_sc, padding="same")

        # Spatial gradients
        grad_y = torch.gradient(flare_contrib, dim=2)[0]
        grad_x = torch.gradient(flare_contrib, dim=3)[0]
        grad_norm_sq = grad_x ** 2 + grad_y ** 2

        # Mean over space and batch
        return grad_norm_sq.mean()

    def compute_loss(self, printed, target, mask, I_diff, I_flare, alpha):
        """
        Total loss:

            L = L_fidelity + λ * L_PV + μ * L_flare
        """
        # Sensitivity and adaptive weights (defined on the mask grid)
        S = self.compute_sensitivity_map(mask, I_diff, I_flare)
        w = self.compute_adaptive_weights(S, alpha)

        # Resize weights to the printed image grid if needed
        if w.shape[-2:] != printed.shape[-2:]:
            w_img = F.interpolate(
                w,
                size=printed.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            w_img = w

        # Weighted fidelity loss
        weighted_l2 = (w_img * (printed - target) ** 2).mean()

        # Process variation band loss (if available)
        pvb_loss = torch.tensor(0.0, device=printed.device)
        if hasattr(self, "compute_pvband"):
            try:
                pvb_loss = self.compute_pvband(mask)
            except Exception:
                # Keep placeholder zero if not implemented
                pvb_loss = torch.tensor(0.0, device=printed.device)

        # Flare regularization
        flare_reg = self.compute_flare_regularization(I_diff)

        total_loss = weighted_l2 + self.lambda1 * pvb_loss + self.mu * flare_reg

        metrics = {
            "S": S,
            "w": w,
            "weighted_l2": weighted_l2,
            "flare_reg": flare_reg,
            "pvb_loss": pvb_loss,
            "total_loss": total_loss,
        }
        return total_loss, metrics

    def _diffraction_limited_image(self, mask):
        """
        Helper to obtain I_diff from the lithography simulator.
        """
        if hasattr(self.litho, "abbe") and callable(self.litho.abbe):
            # Expect mask [B, 1, H, W]; use single-sample Abbe call and
            # restore batch/channel dimensions.
            if mask.dim() == 4 and mask.size(0) == 1 and mask.size(1) == 1:
                I2d = self.litho.abbe(mask[0, 0])
                if I2d.dim() == 2:
                    I2d = I2d.unsqueeze(0).unsqueeze(0)
                return I2d
            return self.litho.abbe(mask)

        # Fallback: use the nominal printed image as a proxy for I_diff
        out = self.litho(mask)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def solve(self, target, params):
        """
        Main flare-aware ILT optimization loop.

        Parameters
        ----------
        target : torch.Tensor
            Target contour / pattern [H, W] or [1, 1, H, W].
        params : torch.Tensor
            Initial mask parameters [H, W] or [1, 1, H, W].

        Returns
        -------
        best_params : torch.Tensor
            Optimized parameter tensor (same spatial shape as input).
        best_mask : torch.Tensor
            Optimized mask tensor in [0, 1] (same spatial shape).
        """
        # Ensure rank-4 tensors [B, 1, H, W]
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        if params.dim() == 2:
            params = params.unsqueeze(0).unsqueeze(0)

        device = next(self.litho.parameters(), target.new_zeros(1)).device
        target = target.to(device)
        params = params.to(device).detach().clone().requires_grad_(True)

        optimizer = Adam([params], lr=self.lr)

        best_loss = None
        best_params = None
        best_mask = None

        bar_width = 30

        for iteration in range(self.max_iters):
            optimizer.zero_grad()

            # 1. Mask from continuous parameters (sigmoid)
            mask = torch.sigmoid(self.theta_M * params)

            # 2. Diffraction-limited image from litho model
            I_diff = self._diffraction_limited_image(mask)

            # 3. Flare contribution I_flare = PSF_SC * I_diff
            psf_sc = self.psf_sc.to(I_diff.device)
            I_flare = F.conv2d(I_diff, psf_sc, padding="same")

            # 4. Flare intensity map alpha(r) at mask resolution
            alpha_mask = self.flare_psf.compute_flare_intensity_map(mask)

            # Separate version of alpha resized to the imaging grid used by
            # I_diff / I_flare for the forward model only.
            if alpha_mask.shape[-2:] != I_diff.shape[-2:]:
                alpha_img = F.interpolate(
                    alpha_mask,
                    size=I_diff.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                alpha_img = alpha_mask

            # 5. Total image with flare: I = I_diff + α · I_flare
            I_total = I_diff + alpha_img * I_flare

            # 6. Simple resist model
            printed = torch.sigmoid(10.0 * (I_total - 0.3))

            # 7. Loss (use alpha at mask resolution for weighting). If the
            # imaging grid resolution differs from the target resolution,
            # resize the target to the printed image grid for the fidelity term.
            if printed.shape[-2:] != target.shape[-2:]:
                target_img = F.interpolate(
                    target,
                    size=printed.shape[-2:],
                    mode="nearest",
                )
            else:
                target_img = target

            loss, metrics = self.compute_loss(printed, target_img, mask, I_diff, I_flare, alpha_mask)
            loss.backward()
            optimizer.step()

            if best_loss is None or loss.item() < best_loss:
                best_loss = loss.item()
                best_params = params.detach().clone()
                best_mask = mask.detach().clone()

            # Text progress bar
            progress = (iteration + 1) / max(self.max_iters, 1)
            filled = int(bar_width * progress)
            bar = "#" * filled + "-" * (bar_width - filled)
            sys.stdout.write(
                f"\r[FlareAwareILT] [{bar}] "
                f"{iteration + 1}/{self.max_iters} "
                f"Loss={loss.item():.4e}"
            )
            sys.stdout.flush()

        # Newline after progress bar completes
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Squeeze back to [H, W] for convenience
        best_params = best_params.squeeze(0).squeeze(0)
        best_mask = best_mask.squeeze(0).squeeze(0)
        return best_params, best_mask

