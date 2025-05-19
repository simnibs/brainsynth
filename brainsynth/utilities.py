import torch


def apply_affine(affine: torch.Tensor, data: torch.Tensor):
    return data @ affine[..., :3, :3].mT + affine[..., :3, [3]].mT
