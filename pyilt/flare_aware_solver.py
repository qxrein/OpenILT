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
        self.bin_weight = self.cfg.get("bin_weight", 0.0)
        self.lr = self.cfg.get("lr", 0.01)
        self.max_iters = int(self.cfg.get("max_iters", 800))
        self.theta_M = self.cfg.get("theta_M", 10.0)
        self.sensitivity_interval = int(self.cfg.get("sensitivity_interval", 5))
        self.use_amp = bool(self.cfg.get("use_amp", True))  # FP16 on GPU for speed

        # Pre-compute PSF_SC kernel (moved to device in solve loop)
        self.psf_sc = self.flare_psf.generate_shiraishi_psf()

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

    def compute_binarization_loss(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Penalize mid-gray values: L_bin = mean(mask * (1 - mask)).
        Minimum when mask is 0 or 1.
        """
        return (mask * (1.0 - mask)).mean()

    def compute_flare_regularization(self, I_diff):
        """
        Flare regularization: L_flare = ∫ ||∇[PSF_SC * I_diff]||² dr
        Uses finite differences (faster than torch.gradient).
        """
        flare_contrib = F.conv2d(I_diff, self.psf_sc.to(I_diff.device), padding="same")
        # Finite differences (4D: pad is (W_left, W_right, H_left, H_right))
        gx = flare_contrib[:, :, :, 1:] - flare_contrib[:, :, :, :-1]
        gy = flare_contrib[:, :, 1:, :] - flare_contrib[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
        gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
        return (gx**2 + gy**2).mean()

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

    def solve(self, target, params, return_history: bool = False):
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target = target.to(device)
        params = params.to(device).detach().clone().requires_grad_(True)
        self.psf_sc = self.psf_sc.to(device)

        optimizer = Adam([params], lr=self.lr)
        best_loss = None
        best_params = None
        best_mask = None
        best_snapshot = {}
        S_cached = None

        history = {
            "loss": [],
            "weighted_l2": [],
            "flare_reg": [],
            "grad_norm": [],
        }
        if self.bin_weight > 0:
            history["bin_loss"] = []
        bar_width = 30
        autocast = torch.autocast("cuda", enabled=(device.type == "cuda" and self.use_amp))

        for iteration in range(self.max_iters):
            optimizer.zero_grad(set_to_none=True)

            with autocast:
                mask = torch.sigmoid(self.theta_M * params)
                I_diff = self._diffraction_limited_image(mask)
                I_flare = F.conv2d(I_diff, self.psf_sc, padding="same")
                alpha_mask = self.flare_psf.compute_flare_intensity_map(mask)

                if alpha_mask.shape[-2:] != I_diff.shape[-2:]:
                    alpha_img = F.interpolate(
                        alpha_mask, size=I_diff.shape[-2:], mode="bilinear", align_corners=False
                    )
                else:
                    alpha_img = alpha_mask

                I_total = I_diff + alpha_img * I_flare
                printed = torch.sigmoid(10.0 * (I_total - 0.3))

                if printed.shape[-2:] != target.shape[-2:]:
                    target_img = F.interpolate(target, size=printed.shape[-2:], mode="nearest")
                else:
                    target_img = target

                # Sensitivity map: recompute every N iters to save cost
                use_cached = (
                    S_cached is not None
                    and iteration > 0
                    and iteration % self.sensitivity_interval != 0
                )
                if use_cached:
                    S = S_cached
                else:
                    S = self.compute_sensitivity_map(mask, I_diff, I_flare)
                    S_cached = S

                w = self.compute_adaptive_weights(S, alpha_mask)
                if w.shape[-2:] != printed.shape[-2:]:
                    w_img = F.interpolate(w, size=printed.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    w_img = w
                weighted_l2 = (w_img * (printed - target_img) ** 2).mean()
                flare_reg = self.compute_flare_regularization(I_diff)
                bin_loss = self.compute_binarization_loss(mask)
                loss = weighted_l2 + self.mu * flare_reg + self.bin_weight * bin_loss
                metrics = {
                    "weighted_l2": weighted_l2,
                    "flare_reg": flare_reg,
                    "bin_loss": bin_loss,
                }
            loss.backward()

            grad_norm = 0.0
            if params.grad is not None:
                grad_norm = params.grad.data.norm(2).item()
            history["grad_norm"].append(grad_norm)

            optimizer.step()

            history["loss"].append(float(loss.detach().cpu()))
            history["weighted_l2"].append(float(metrics["weighted_l2"].detach().cpu()))
            history["flare_reg"].append(float(metrics["flare_reg"].detach().cpu()))
            if self.bin_weight > 0:
                history["bin_loss"].append(float(metrics["bin_loss"].detach().cpu()))

            if best_loss is None or loss.item() < best_loss:
                best_loss = loss.item()
                best_params = params.detach().clone()
                best_mask = mask.detach().clone()
                # Store a lightweight snapshot for paper-friendly outputs
                best_snapshot = {
                    "printed": printed.detach().clone(),
                    "I_diff": I_diff.detach().clone(),
                    "alpha_mask": alpha_mask.detach().clone(),
                    "S": S.detach().clone(),
                    "w": w.detach().clone(),
                }

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

        if return_history:
            return best_params, best_mask, {"history": history, "best_snapshot": best_snapshot}
        return best_params, best_mask

