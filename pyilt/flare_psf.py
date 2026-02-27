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

        # Radial interpolation: nearest neighbor over the sampled 1D profile
        rr_flat = rr.view(-1)
        kernel_flat = torch.empty_like(rr_flat)
        for idx in range(rr_flat.numel()):
            ridx = torch.argmin(torch.abs(r - rr_flat[idx]))
            kernel_flat[idx] = psf_values[ridx]
        kernel = kernel_flat.view_as(rr)

        # Normalize PSF_SC to unit sum
        kernel = kernel / kernel.sum()

        # Shape: [1, 1, H, W] for conv2d
        return kernel.unsqueeze(0).unsqueeze(0)

    def compute_flare_intensity_map(self, mask: torch.Tensor, sigma_nm: float = 2000.0) -> torch.Tensor:
        """
        Compute the spatially varying flare intensity map

            alpha(r) = gamma * [K_density * M](r)

        where K_density is a Gaussian kernel with standard deviation sigma_nm.

        Parameters
        ----------
        mask : torch.Tensor
            Mask tensor of shape [B, 1, H, W].
        sigma_nm : float
            Standard deviation of the Gaussian kernel in nm (default 2 µm).

        Returns
        -------
        alpha : torch.Tensor
            Flare intensity map with the same shape as `mask`.
        """
        if mask.dim() != 4 or mask.size(1) != 1:
            raise ValueError(f"mask must have shape [B, 1, H, W], got {mask.shape}")

        device = mask.device

        # Convert sigma from nm to pixels
        sigma_px = sigma_nm / float(self.pixel_size)

        # Kernel size ~ 6 sigma (3-sigma on each side)
        kernel_size = int(6 * sigma_px)
        if kernel_size % 2 == 0:
            kernel_size += 1

        half = kernel_size // 2
        x = torch.arange(-half, half + 1, device=device, dtype=mask.dtype)
        xx, yy = torch.meshgrid(x, x, indexing="ij")

        gaussian = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_px ** 2))
        gaussian = gaussian / gaussian.sum()
        gaussian = gaussian.unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]

        # Convolution over the mask gives a local pattern density
        density = F.conv2d(mask, gaussian, padding=half)

        # Calibrated gain factor for alpha(r)
        gamma = 0.1
        alpha = gamma * density

        return alpha

