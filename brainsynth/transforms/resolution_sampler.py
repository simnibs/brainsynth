from collections.abc import Sequence

import torch

from .base import BaseTransform


class ResolutionSamplerDefault(BaseTransform):
    def __init__(self, device: None | torch.device = None):
        super().__init__(device)
        self.clinical_low_res = 6.0     # 2 : 2.5-4.5
        self.low_field_stock = 5.0      # 3 :
        self.low_field_iso = 3.0        # 1

    def forward(self):
        with torch.device(self.device):
            r = torch.rand(1)
            # if r < 0.25:  # 1mm isotropic
            #     resolution = torch.tensor([1.0, 1.0, 1.0])
            #     thickness = torch.tensor([1.0, 1.0, 1.0])
            if r < 0.33:
                # clinical (low-res in one dimension)
                resolution = torch.tensor([1.0, 1.0, 1.0])
                thickness = torch.tensor([1.0, 1.0, 1.0])
                idx = torch.randint(0, 3, (1,))
                resolution[idx] = 2.5 + self.clinical_low_res * torch.rand(1)
                thickness[idx] = torch.minimum(
                    resolution[idx], 4.0 + 2.0 * torch.rand(1)
                )
            elif r < 0.66:
                # low-field: stock sequences (always axial)
                resolution = torch.tensor([1.3, 1.3, self.low_field_stock]) + 0.4 * torch.rand(3)
                thickness = resolution.clone()
            else:
                # low-field: isotropic-ish (also good for scouts)
                resolution = 2.0 + self.low_field_iso * torch.rand(3)
                thickness = resolution.clone()

        return resolution, thickness


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
