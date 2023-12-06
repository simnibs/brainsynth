import torch


def resolution_sampler(device):
    with torch.device(device):
        r = torch.rand(1)
        if r < 0.25: # 1mm isotropic
            resolution = torch.Tensor([1.0, 1.0, 1.0])
            thickness = torch.Tensor([1.0, 1.0, 1.0])
        elif r < 0.5: # clinical (low-res in one dimension)
            resolution = torch.Tensor([1.0, 1.0, 1.0])
            thickness = torch.Tensor([1.0, 1.0, 1.0])
            idx = torch.randint(0, 3, (1,))
            resolution[idx] = 2.5 + 6 * torch.rand(1)
            thickness[idx] = torch.minimum(resolution[idx], 4.0 + 2.0 * torch.rand(1))
        elif r < 0.75:  # low-field: stock sequences (always axial)
            resolution = torch.Tensor([1.3, 1.3, 5.0]) + 0.4 * torch.rand(3)
            thickness = resolution.clone()
        else: # low-field: isotropic-ish (also good for scouts)
            resolution = 2.0 + 3.0 * torch.rand(3)
            thickness = resolution.clone()
        return resolution, thickness


def make_affine_matrix(rot, shear, scale, device):
    Rx = torch.Tensor([[1, 0, 0], [0, torch.cos(rot[0]), -torch.sin(rot[0])], [0, torch.sin(rot[0]), torch.cos(rot[0])]], device=device)
    Ry = torch.Tensor([[torch.cos(rot[1]), 0, torch.sin(rot[1])], [0, 1, 0], [-torch.sin(rot[1]), 0, torch.cos(rot[1])]], device=device)
    Rz = torch.Tensor([[torch.cos(rot[2]), -torch.sin(rot[2]), 0], [torch.sin(rot[2]), torch.cos(rot[2]), 0], [0, 0, 1]], device=device)

    SHx = torch.Tensor([[1, 0, 0], [shear[1], 1, 0], [shear[2], 0, 1]], device=device)
    SHy = torch.Tensor([[1, shear[0], 0], [0, 1, 0], [0, shear[2], 1]], device=device)
    SHz = torch.Tensor([[1, 0, shear[0]], [0, 1, shear[1]], [0, 0, 1]], device=device)

    A = SHx @ SHy @ SHz @ Rx @ Ry @ Rz
    A[0] *= scale[0]
    A[1] *= scale[1]
    A[2] *= scale[2]

    return A
