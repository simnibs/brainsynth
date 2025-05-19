from collections.abc import Sequence

import torch

from .base import BaseTransform


class RandClinicalSlice(BaseTransform):
    def __init__(self, low: float = 2.5, high: float = 8.5, slice_idx: int | None  = None, device: None | torch.device = None):
        super().__init__(device)
        self.slice_idx = slice_idx
        with torch.device(self.device):
            self.uniform = torch.distributions.Uniform(low, high)

    def forward(self):
        # clinical (low-res in one dimension)
        with torch.device(self.device):
            resolution = torch.tensor([1.0, 1.0, 1.0])
            thickness = torch.tensor([1.0, 1.0, 1.0])

            idx = torch.randint(0, 3, (1,)) if self.slice_idx is None else self.slice_idx

            resolution[idx] = self.uniform.sample()
            thickness[idx] = torch.minimum(
                resolution[idx], 4.0 + 2.0 * torch.rand(1)
            )
        return resolution, thickness

class RandLowFieldStock(BaseTransform):
    def __init__(self, res_base = (1.3, 1.3, 5.0), device: None | torch.device = None):
        """Random low-field stock sequence (always axial)."""
        super().__init__(device)
        with torch.device(self.device):
            self.res_base = torch.tensor(res_base)
            self.uniform = torch.distributions.Uniform(0.0, 0.4)

    def forward(self):
        resolution = self.res_base + self.uniform.sample((3,))
        thickness = resolution.clone()
        return resolution, thickness


class RandLowFieldIsotropic(BaseTransform):
    """Random low-field isotropic-ish scan."""
    def __init__(self, low: float = 2.0, high: float = 5.0, device: None | torch.device = None):
        super().__init__(device)
        with torch.device(self.device):
            self.uniform = torch.distributions.Uniform(low, high)

    def forward(self):
        resolution = self.uniform.sample((3,))
        thickness = resolution.clone()
        return resolution, thickness

class ResolutionSamplerDefault(BaseTransform):
    def __init__(self, device: None | torch.device = None):
        super().__init__(device)
        # self.clinical = RandClinicalSlice(5.99, 6.01, device=self.device)
        self.clinical = RandClinicalSlice(device=self.device)
        self.low_field_stock = RandLowFieldStock(device=self.device)
        self.low_field_iso = RandLowFieldIsotropic(device=self.device)

    def forward(self):
        # return self.clinical()
        r = torch.rand(1, device=self.device)
        if r < 0.33:
            return self.clinical()
        elif r < 0.66:
            return self.low_field_stock()
        else:
            return self.low_field_iso()


class ResolutionSamplerPhoto(BaseTransform):
    def __init__(
        self,
        in_res: Sequence,
        spacing: float,
        slice_thickness: float = 0.001,
        device: None | torch.device = None,
    ):
        super().__init__(device)
        self.in_res = torch.as_tensor(in_res, device=device)
        self.spacing = spacing
        self.slice_thickness = slice_thickness

    def forward(self):
        with torch.device(self.device):
            out_res = torch.tensor([self.in_res[0], self.spacing, self.in_res[2]])
            thickness = torch.tensor(
                [self.in_res[0], self.slice_thickness, self.in_res[2]]
            )
        return out_res, thickness
