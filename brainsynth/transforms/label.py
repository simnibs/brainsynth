import torch

class Reindex(torch.nn.Module):
    def __init__(self, labels: torch.IntTensor):
        """Reindex data, e.g., [1,3,5,7] to [0,1,2,3]."""
        kwargs = dict(dtype=torch.int, device=labels.device)
        labels = labels.sort().values
        self.reindexer = torch.zeros(labels.amax() + 1, **kwargs)
        self.reindexer[labels] = torch.arange(len(labels), **kwargs)

    def forward(self, data: torch.IntTensor):
        return self.reindexer[data]


class AsDiscreteWithReindex(torch.nn.Module):
    def __init__(self, labels) -> None:
        super().__init__(
            [
                monai.transforms.EnsureType(dtype=torch.int),
                Reindex(labels),
                monai.transforms.AsDiscrete(to_onehot=len(labels)),
            ]
        )

    def forward(self, image):
