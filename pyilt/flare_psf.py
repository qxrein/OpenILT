import torch
import torch.nn.functional as F


class FlarePSF:
    """
    Flare PSF utilities following a Shiraishi-style formulation.

    Notes
    -----
    - The current `generate_shiraishi_psf` is a placeholder and should be
      replaced with a calibrated PSF_SC(r) from Shiraishi et al.
    - All computations are implemented in PyTorch so they are differentiable
      and run on the same device as the mask tensors.
    """

    def __init__(self, tis: float = 0.08, kernel_size: int = 401, pixel_size: float = 1.0):
        """
        Parameters
        ----------
        tis : float
            Total Integrated Scatter (TIS).
        kernel_size : int
            Size of the PSF kernel (must be odd).
        pixel_size : float
            Pixel size in nm.
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number.")
        self.tis = tis
        self.kernel_size = kernel_size
        self.pixel_size = pixel_size

    def generate_shiraishi_psf(self, radius_um: float = 4.0) -> torch.Tensor:
        """
        Generate a placeholder Shiraishi-style scattering PSF (PSF_SC).

        This should be replaced with the actual PSF profile from Shiraishi's
        paper or from measured data.

        Returns
        -------
        kernel : torch.Tensor
            2D PSF_SC kernel with shape [1, 1, H, W].
        """
        # Maximum radius in nm for the radial profile
        r_max_nm = radius_um * 1000.0

        # 1D radial coordinate and radial PSF profile (placeholder form)
        r = torch.linspace(0.0, r_max_nm, self.kernel_size)
        # Avoid division by zero at r = 0
        psf_values = 1.0 / (r ** 2 + 100.0)

        # 2D coordinates in nm
        x = torch.linspace(-r_max_nm, r_max_nm, self.kernel_size)
        y = torch.linspace(-r_max_nm, r_max_nm, self.kernel_size)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        rr = torch.sqrt(xx ** 2 + yy ** 2)

        # Radial interpolation: vectorized nearest neighbor
        rr_flat = rr.view(-1)
        # r is [0, r_max]; for each rr find closest r index
        ridx = torch.clamp(
            ((rr_flat / r_max_nm) * (self.kernel_size - 1)).long(),
            0, self.kernel_size - 1
        )
        kernel_flat = psf_values[ridx]
        kernel = kernel_flat.view_as(rr)

        # Normalize PSF_SC to unit sum
        kernel = kernel / kernel.sum()

        # Shape: [1, 1, H, W] for conv2d
        return kernel.unsqueeze(0).unsqueeze(0)

    def compute_flare_intensity_map(
        self, mask: torch.Tensor, sigma_nm: float = 2000.0, max_kernel_size: int = 65
    ) -> torch.Tensor:
        """
        Compute the spatially varying flare intensity map

            alpha(r) = gamma * [K_density * M](r)

        where K_density is a Gaussian kernel.

        Parameters
        ----------
        mask : torch.Tensor
            Mask tensor of shape [B, 1, H, W].
        sigma_nm : float
            Standard deviation in nm (2 µm default).
        max_kernel_size : int
            Cap kernel size for speed (default 65).

        Returns
        -------
        alpha : torch.Tensor
            Flare intensity map with the same shape as `mask`.
        """
        if mask.dim() != 4 or mask.size(1) != 1:
            raise ValueError(f"mask must have shape [B, 1, H, W], got {mask.shape}")

        sigma_px = max(sigma_nm / float(self.pixel_size), 2.0)
        kernel_size = min(int(6 * sigma_px), max_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        half = kernel_size // 2
        x = torch.arange(-half, half + 1, device=mask.device, dtype=mask.dtype)
        xx, yy = torch.meshgrid(x, x, indexing="ij")
        gaussian = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma_px**2))
        gaussian = (gaussian / gaussian.sum()).unsqueeze(0).unsqueeze(0)
        density = F.conv2d(mask, gaussian, padding=half)
        return 0.1 * density

