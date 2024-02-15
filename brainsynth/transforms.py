import monai
import torch


class Reindex(monai.transforms.Transform):
    def __init__(self, labels: torch.IntTensor):
        """Reindex data, e.g., [1,3,5,7] to [0,1,2,3]."""
        kwargs = dict(dtype=torch.int, device=labels.device)
        labels = labels.sort().values
        self.reindexer = torch.zeros(labels.amax() + 1, **kwargs)
        self.reindexer[labels] = torch.arange(len(labels), **kwargs)

    def __call__(self, data: torch.IntTensor):
        return self.reindexer[data]


class AsDiscreteWithReindex(monai.transforms.Compose):
    def __init__(self, labels) -> None:
        super().__init__(
            [
                monai.transforms.EnsureType(dtype=torch.int),
                Reindex(labels),
                monai.transforms.AsDiscrete(to_onehot=len(labels)),
            ]
        )


class SpatialCrop(monai.transforms.Transform):
    def __init__(
            self,
            in_size: torch.Tensor,
            out_size: torch.Tensor,
            out_center: torch.Tensor,
        ) -> None:
        self.in_size = in_size
        self.out_size = out_size
        self.out_center = out_center

        halfsize = 0.5 * out_size
        start = torch.ceil(out_center - halfsize).int()
        stop = torch.floor(out_center + halfsize).int()
        stop += out_size - (stop-start)
        slices = tuple((a.item(), b.item()) for a,b in zip(start, stop))

        # sample these slices in the original image
        self.slice_in = tuple(
            slice(max(slc[0], 0), min(slc[1], ins.item())) for slc,ins in zip(slices, in_size)
        )
        # place them in these positions in the new image
        self.slice_out = tuple(
            slice(0 - min(0, slc[0]), outs.item() - max(0, slc[1]-ins.item())) for slc,ins,outs in zip(slices, in_size, out_size)
        )

    def __call__(self, image: monai.data.MetaTensor) -> monai.data.MetaTensor:
        out_size = torch.zeros(image.ndim, dtype=self.out_size.dtype)
        out_size[-3:] = self.out_size
        for i,s in enumerate(image.shape[:-3]):
            out_size[i] = s

        out = monai.data.MetaTensor(
            torch.zeros(tuple(out_size), dtype=image.dtype), meta=image.meta
        )
        out[..., *self.slice_out] = image[..., *self.slice_in]
        return out

class NormalizeIntensity(monai.transforms.Transform):
    def __call__(self, image):
        r = image.aminmax()
        return (image - r.min) / (r.max - r.min)

# Random transforms

class RandomizableTransform(monai.transforms.Transform):
    """Similar to monai.transforms.RandomizableTransform but using torch."""
    def __init__(self, prob: float = 1.0, do_transform: bool = True):
        self._do_transform = do_transform
        self.prob = min(max(prob, 0.0), 1.0)

    def randomize(self) -> None:
        """
        """
        self._do_transform = torch.rand(1) < self.prob

class RandGaussianNoise(RandomizableTransform):
    def __init__(self, prob, mean, std_range):
        super().__init__(prob)
        self.mean = mean
        self.std_range = std_range

    def __call__(self, image: monai.data.MetaTensor) -> monai.data.MetaTensor:
        super().randomize()
        if not self._do_transform:
            return image
        a,b = self.std_range
        std = a + (b - a) * torch.rand(1)
        return image + self.mean + std * torch.randn(image.shape)

class RandGammaTransform(RandomizableTransform):
    def __init__(self, prob: float, mean, std):
        super().__init__(prob)
        self.mean = mean
        self.std = std

    def _sample_gamma(self):
        return torch.exp(self.mean + self.std * torch.randn(1))

    def __call__(self, image: monai.data.MetaTensor) -> monai.data.MetaTensor:
        super().randomize()
        if not self._do_transform:
            return image
        gamma = self._sample_gamma()
        # apply transform
        ra = image.aminmax()
        r = ra.max - ra.min
        return ((image - ra.min) / r) ** gamma * r + ra.min
