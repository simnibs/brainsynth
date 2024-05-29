import torch

from .base import BaseTransform


class GaussianSmooth(BaseTransform):
    def __init__(
        self,
        sigma: float | list[float] | tuple,
        tail_length: float = 4.0,
        spatial_dims=3,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)

        self.spatial_dims = spatial_dims
        self.tail_length = tail_length

        sigma = (
            tuple(float(sigma) for _ in range(spatial_dims))
            if isinstance(sigma, (float, int))
            else sigma
        )
        assert len(sigma) == spatial_dims

        self.kernels = self._compute_gaussian_kernels(sigma, tail_length)
        self.conv = getattr(torch.nn.functional, f"conv{spatial_dims}d")

    def _compute_gaussian_kernels(self, sigma: list | tuple, tail_length: float):
        return tuple(
            self._compute_gaussian_kernel(s, tail_length, dim=i)
            if s > 0.0 else None for i, s in enumerate(sigma)
        )

    def _compute_gaussian_kernel(self, s: float, tail_length: float, dim: int):
        tail = int(max(round(s * tail_length), 1.0))
        x = torch.arange(-tail, tail + 1)

        N = torch.distributions.Normal(0.0, s)
        kernel = N.log_prob(x).exp()
        kernel /= kernel.sum()

        # reshape
        _dim = dim + 2  # spatial dim + batch + channel
        _dim_total = self.spatial_dims + 2
        return kernel.reshape(*[-1 if j == _dim else 1 for j in range(_dim_total)])

    def forward(self, x):
        for kernel in self.kernels:
            if kernel is not None: # sigma = 0.0
                x = self.conv(x, kernel, padding="same")
        return x

        # image = nib.load("/mrhome/jesperdn/nobackup/ernie/ernie_T1.nii.gz").get_fdata()
        # img = torch.from_numpy(image[None]).float()
        # g = GaussianSmooth(1.0)(img)
        # w = monai.transforms.GaussianSmooth(1.0)(torch.from_numpy(image[None]).float())
