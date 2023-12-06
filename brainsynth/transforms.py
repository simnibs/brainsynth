import monai
import torch

class Reindex(monai.transforms.Transform):
    def __init__(self, labels: torch.Tensor):
        """Reindex data, e.g., [1,3,5,7] to [0,1,2,3].
        """
        labels = labels.sort().values
        self.reindexer = torch.zeros(labels.amax() + 1, dtype=torch.int)
        self.reindexer[labels] = torch.arange(len(labels), dtype=torch.int)

    def __call__(self, data: torch.IntTensor):
        return self.reindexer[data]

