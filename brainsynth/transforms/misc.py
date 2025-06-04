from typing import Any, Callable, Sequence

import torch

from .base import BaseTransform


class Uniform(BaseTransform):
    def __init__(self, low=0.0, high=1.0, device: None | torch.device = None):
        super().__init__(device)
        self.low = low
        self.high = high

    def forward(self, shape=torch.Size([])):
        return self.low + self.high * torch.rand(shape, device=self.device)


class ExtractDictKeys(BaseTransform):
    def __init__(self, device: None | torch.device = None) -> None:
        super().__init__(device)

    def forward(self, d: dict):
        return tuple(d.keys())


class ExtractDictValues(BaseTransform):
    def __init__(self, device: None | torch.device = None) -> None:
        super().__init__(device)

    def forward(self, d: dict):
        return tuple(d.values())


class SelectEntry(BaseTransform):
    def __init__(
        self, index: int | None = None, device: None | torch.device = None
    ) -> None:
        super().__init__(device)
        self.index = index

    def forward(self, x):
        return x[self.index]


class AssertCondition(BaseTransform):
    def __init__(
        self,
        assert_func: Callable,
        assert_fail_message: str = "",
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        self.assert_func = assert_func
        self.assert_fail_message = assert_fail_message

    def forward(self, v):
        assert self.assert_func(v), self.assert_fail_message
        return v


class Intersection(BaseTransform):
    def __init__(
        self, elements: Sequence[Any], device: None | torch.device = None
    ) -> None:
        super().__init__(device)
        if len(elements) == 0:
            raise ValueError(
                f"Cannot compute intersection when `elements` is empty (elements = {elements})."
            )
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
