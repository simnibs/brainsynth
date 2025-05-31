import torch


def apply_affine(affine: torch.Tensor, data: torch.Tensor):
    return data @ affine[..., :3, :3].mT + affine[..., :3, [3]].mT


def squeeze_nd(t: torch.Tensor, n: int, dim: int = 0):
    return squeeze_nd(t.squeeze(dim=dim), n, dim) if t.ndim > n else t


def unsqueeze_nd(tensor, n):
    return unsqueeze_nd(tensor[None], n) if tensor.ndim < n else tensor
