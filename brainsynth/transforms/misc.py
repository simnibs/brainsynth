from typing import Any,Sequence

import torch

from .base import BaseTransform

class Uniform(BaseTransform):
    def __init__(self, low=0.0, high=1.0, device: None | torch.device = None):
        super().__init__(device)
        self.low = low
        self.high = high

    def forward(self, shape = torch.Size([])):
        return self.low + self.high * torch.rand(shape, device=self.device)

class ExtractDictKeys(BaseTransform):
    def __init__(self, device: None | torch.device = None) -> None:
        super().__init__(device)

    def forward(self, d: dict):
        return tuple(d.keys())

class Intersection(BaseTransform):
    def __init__(self, elements: Sequence[Any], device: None | torch.device = None) -> None:
        super().__init__(device)
        if len(elements) == 0:
            raise ValueError(f"Cannot compute intersection when `elements` is empty (elements = {elements}).")
        self.elements = elements

    def forward(self, test_elements):
        return tuple(i for i in self.elements if i in test_elements)

class ServeValue(BaseTransform):
    def __init__(self, x, device: None | torch.device = None) -> None:
        super().__init__(device)
        self.x = x

    def forward(self):
        return self.x

class ApplyFunction(BaseTransform):
    def __init__(self, func, device: None | torch.device = None) -> None:
        super().__init__(device)
        self.func = func

    def forward(self, x):
        return self.func(x)
