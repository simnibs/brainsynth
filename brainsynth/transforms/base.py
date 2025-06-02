import torch


class BaseTransform(torch.nn.Module):
    def __init__(self, device: None | torch.device = None) -> None:
        super().__init__()
        self.device = torch.device("cpu") if device is None else device


class IdentityTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SequentialTransform(BaseTransform):
    def __init__(self, *transforms):
        super().__init__()
        self.transforms = tuple(transforms)

    def forward(self, x=None):
        for transform in self.transforms:
            x = transform() if x is None else transform(x)
        return x


class RandomizableTransform(BaseTransform):
    """Similar to monai.transforms.RandomizableTransform but using torch."""

    def __init__(self, prob: float = 1.0, device: None | torch.device = None):
        super().__init__(device)
        assert 0 <= prob <= 1.0, "Invalid probability"
        self.prob = prob
        self.randomize()

    def randomize(self) -> None:
        """ """
        self._do_transform = torch.rand(1, device=self.device) < self.prob

    def should_apply_transform(self):
        return self._do_transform


class RandomChoice(BaseTransform):
    def __init__(
        self,
        args: tuple | list,
        prob: tuple | list | None = None,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        n = len(args)
        if prob is None:
            self.prob = torch.full((n,), 1 / n, device=device)
        else:
            self.prob = torch.tensor(prob, device=device)
        assert self.prob.sum() == 1.0
        assert len(args) == len(self.prob)
        self.choices = args

    def forward(self):
        return self.choices[torch.multinomial(self.prob, 1)]


class SwitchTransform(BaseTransform):
    def __init__(
        self,
        *transforms,
        prob: tuple | list | None = None,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        n = len(transforms)
        if prob is None:
            self.prob = torch.full((n,), 1 / n, device=device)
        else:
            self.prob = torch.tensor(prob, device=device)
        assert self.prob.sum() == 1.0
        assert len(transforms) == len(self.prob)
        self.transforms = transforms

    def forward(self, x):
        return self.transforms[torch.multinomial(self.prob, 1)](x)


class IfTransform(BaseTransform):
    def __init__(self, condition, *transforms, device: None | torch.device = None):
        super().__init__(device)
        self.condition = condition
        self.transforms = transforms

    def forward(self, x):
        if self.condition:
            return self.transforms(x)


class EnsureDevice(BaseTransform):
    def __init__(self, device: None | torch.device = None) -> None:
        super().__init__(device)

    def forward(self, x: dict | list | torch.Tensor | tuple):
        if isinstance(x, torch.Tensor):
            return x.to(self.device, non_blocking=True)
        elif isinstance(x, (tuple, list)):
            return [self.forward(i) for i in x]
        elif isinstance(x, dict):
            return {k: self.forward(v) for k, v in x.items()}


class EnsureDType(BaseTransform):
    def __init__(self, dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, x: dict | list | torch.Tensor | tuple):
        if isinstance(x, torch.Tensor):
            return x.to(self.dtype, non_blocking=True)
        elif isinstance(x, (tuple, list)):
            return [self.forward(i) for i in x]
        elif isinstance(x, dict):
            return {k: self.forward(v) for k, v in x.items()}
