import torch

class TransformPipeline(torch.nn.Module):
    def __init__(self, transforms):
        self.transforms = transforms

    def forward(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image

class RandomizableTransform(torch.nn.Module):
    """Similar to monai.transforms.RandomizableTransform but using torch."""
    def __init__(self, prob: float = 1.0):
        self._do_transform = None
        self.prob = min(max(prob, 0.0), 1.0)
        self.randomize()

    def randomize(self) -> None:
        """
        """
        self._do_transform = torch.rand(1) < self.prob