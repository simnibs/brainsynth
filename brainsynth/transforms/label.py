import torch

from .base import BaseTransform


# class Reindex(BaseTransform):
#     def __init__(self, labels: torch.IntTensor):
#         """Reindex data, e.g., [1,3,5,7] to [0,1,2,3]."""
#         kwargs = dict(dtype=torch.int, device=labels.device)
#         labels = labels.sort().values
#         self.reindexer = torch.zeros(labels.amax() + 1, **kwargs)
#         self.reindexer[labels] = torch.arange(len(labels), **kwargs)

#     def forward(self, data: torch.IntTensor):
#         return self.reindexer[data]


# class AsDiscreteWithReindex(torch.nn.Module):
#     def __init__(self, labels) -> None:
#         super().__init__(
#             [
#                 EnsureDType(torch.int),
#                 Reindex(labels),
#                 monai.transforms.AsDiscrete(to_onehot=len(labels)),
#             ]
#         )

#     def forward(self, image):
#         pass


class MaskFromLabelImage(BaseTransform):
    def __init__(self, labels: list | tuple | torch.Tensor, device: None | torch.device = None) -> None:
        super().__init__(device)
        self.labels = torch.as_tensor(labels, device=self.device)

    def forward(self, label_image: torch.Tensor):
        return torch.isin(label_image, self.labels)

class OneHotEncoding(BaseTransform):
    def __init__(self, num_classes: int, device: None | torch.device = None) -> None:
        """This transform assumes that the values of the label tensor is 0 to
        (num_classes-1).
        """
        super().__init__(device)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        assert not x.is_floating_point()
        spatial_shape = x.shape[-3:]
        n = x.numel()
        out = torch.zeros((self.num_classes, n), dtype=x.dtype, device=x.device)
        out[x.ravel(), torch.arange(n, dtype=torch.int)] = 1
        return out.reshape(self.num_classes, *spatial_shape)
