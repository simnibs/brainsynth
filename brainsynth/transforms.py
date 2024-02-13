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

