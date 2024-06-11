import warnings

import torch

from brainsynth.transforms.base import BaseTransform, RandomizableTransform
from brainsynth.transforms import resolution_sampler
from brainsynth.transforms.filters import GaussianSmooth


# import functools

# def dict_recursion(func):
#     @functools.wraps(func)
#     def wrapper(x):
#         if isinstance(x, dict):
#             return {k:wrapper(v) for k,v in x.items()}
#         else:
#             return func(x)
#     return wrapper


class SpatialCrop(BaseTransform):
    def __init__(
        self,
        in_size: torch.Tensor,
        out_size: torch.Tensor,
        out_center: torch.Tensor,
    ) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.out_center = out_center

        halfsize = 0.5 * (out_size - 1)
        start = torch.ceil(out_center - halfsize).int()
        stop = torch.floor(out_center + halfsize).int()
        stop += out_size - (stop - start)
        slices = tuple((a.item(), b.item()) for a, b in zip(start, stop))

        # sample these slices in the original image
        self.slice_in = tuple(
            slice(max(slc[0], 0), min(slc[1], ins.item()))
            for slc, ins in zip(slices, in_size)
        )
        # place them in these positions in the new image
        self.slice_out = tuple(
            slice(0 - min(0, slc[0]), outs.item() - max(0, slc[1] - ins.item()))
            for slc, ins, outs in zip(slices, in_size, out_size)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out_size = torch.zeros(image.ndim, dtype=self.out_size.dtype)
        out_size[-3:] = self.out_size
        for i, s in enumerate(image.shape[:-3]):
            out_size[i] = s

        out = torch.zeros(tuple(out_size), dtype=image.dtype)
        out[..., *self.slice_out] = image[..., *self.slice_in]
        return out


class TranslationTransform(BaseTransform):
    def __init__(
        self,
        translation: torch.Tensor,
        invert: bool = False,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        self.invert = invert
        self.translation = translation.to(self.device)

    def forward(self, x: torch.Tensor):
        if self.invert:
            return x - self.translation
        else:
            return x + self.translation


class RandLinearTransform(RandomizableTransform):
    def __init__(
        self,
        max_rotation,
        max_scale,
        max_shear,
        device: None | torch.device = None,
        prob: float = 1.0,
    ):
        super().__init__(prob, device)

        if not self.should_apply_transform():
            self.trans = torch.eye(3, device=self.device)
            self.trans_inv = self.trans
            self.scale = 1.0
            return

        self.max_scale = max_scale
        self.max_rotation = max_rotation
        self.max_shear = max_shear

        rotations = (
            (2 * self.max_rotation * torch.rand(3) - self.max_rotation)
            / 180.0
            * torch.pi
        )
        shears = 2 * self.max_shear * torch.rand(3) - self.max_shear
        scalings = 1 + (2 * self.max_scale * torch.rand(3) - self.max_scale)
        # we divide distance maps by this, not perfect, but better than nothing
        scaling_factor_distances = torch.prod(scalings) ** 0.33333333333

        self.trans = self.generate_transform(rotations, shears, scalings)
        self.trans_inv = self.inverse()
        self.scale = scaling_factor_distances

    def generate_transform(
        self,
        rot: torch.Tensor,
        shear: torch.Tensor,
        scale: torch.Tensor,
    ):
        with torch.device(self.device):
            cosR = torch.cos(rot)
            sinR = torch.sin(rot)

            # rotation
            Rx = torch.tensor(
                [
                    [1, 0, 0],
                    [0, cosR[0], -sinR[0]],
                    [0, sinR[0], cosR[0]],
                ]
            )
            Ry = torch.tensor(
                [
                    [cosR[1], 0, sinR[1]],
                    [0, 1, 0],
                    [-sinR[1], 0, cosR[1]],
                ]
            )
            Rz = torch.tensor(
                [
                    [cosR[2], -sinR[2], 0],
                    [sinR[2], cosR[2], 0],
                    [0, 0, 1],
                ]
            )

            # shear
            SHx = torch.tensor([[1, 0, 0], [shear[1], 1, 0], [shear[2], 0, 1]])
            SHy = torch.tensor([[1, shear[0], 0], [0, 1, 0], [0, shear[2], 1]])
            SHz = torch.tensor([[1, 0, shear[0]], [0, 1, shear[1]], [0, 0, 1]])

            # collect
            A = SHx @ SHy @ SHz @ Rx @ Ry @ Rz
            A[0] *= scale[0]
            A[1] *= scale[1]
            A[2] *= scale[2]

        return A

    def inverse(self):
        return torch.linalg.inv(self.trans)

    def forward(self, grid, inverse: bool = False):
        """`grid` has coordinates in the last dimension!"""
        if self.should_apply_transform():
            A = self.trans_inv if inverse else self.trans
            return grid @ A.T
        else:
            return grid


class RandNonlinearTransform(RandomizableTransform):
    def __init__(
        self,
        out_size,
        scale_min: float,
        scale_max: float,
        std_max: float,
        exponentiate_field: bool = False,
        scale_and_square_steps: int = 8,
        grid: None | torch.Tensor = None,
        photo_mode: bool = False,
        device: None | torch.device = None,
        prob: float = 1.0,
    ):
        """
        To be consistent with torch convention that the channel dimension is
        before the spatial dimensions, we create the deformation field such
        that the deformations are [C,W,H,D] although when applying it in
        `compute_deformed_grid` we use it as [W,H,D,C].

        compute_backward_field
            Integrate SVF.
        n_steps
            Number of integration steps for

        """
        super().__init__(prob, device)

        self.out_size = out_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.std_max = std_max
        self.grid = grid
        self.exponentiate_field = exponentiate_field
        self.scale_and_square_steps = scale_and_square_steps

        if not self.should_apply_transform():
            self.trans = {}
            return

        nonlin_scale = scale_min + torch.rand(1, device=self.device) * (
            scale_max - scale_min
        )
        size_F_small = torch.round(nonlin_scale * out_size).to(torch.int)

        if photo_mode:
            size_F_small[1] = torch.round(out_size[1] / self.spacing).to(
                size_F_small.dtype
            )
        nonlin_std = std_max * torch.rand(1, device=self.device)

        # Channel first (as is normal)
        V = nonlin_std * torch.randn([3, *size_F_small], device=self.device)
        V = Resize(out_size)(V)

        if photo_mode:
            # no deformation in AP direction
            V[1] = 0

        if exponentiate_field:
            # this is slow on the CPU
            steplength = 1.0 / (2.0**scale_and_square_steps)

            # forward integration
            U = V * steplength
            U = self.scale_and_square(U, self.scale_and_square_steps)

            # backward integration
            U_inv = -V * steplength
            U_inv = self.scale_and_square(U_inv, self.scale_and_square_steps)

            self.trans = dict(forward=U, backward=U_inv)
        else:
            self.trans = dict(forward=V)

    def scale_and_square(self, field, n_steps):
        """Scale and square for matrix exponentiation.

        Parameters
        ----------
        field : _type_
            _description_
        n_steps : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        References
        ----------
        Ashburner (2007). A fast diffeomorphic image registration algorithm.
        Arsigny (2006). A Log-Euclidean Framework for Statistics on
            Diffeomorphisms.
        """
        if n_steps == 0:
            return field
        else:
            resample = GridSample(self.grid + self.channel_last(field), self.out_size)
            field += resample(field)
            return self.scale_and_square(field, n_steps - 1)

    @staticmethod
    def channel_last(x):
        return x.permute((1, 2, 3, 0))

    def forward(self, grid, direction="forward"):
        if self.should_apply_transform():
            match direction:
                case "forward":
                    return grid + self.channel_last(self.trans[direction])
                case "backward":
                    sampler = GridSample(grid, self.out_size)
                    return grid + sampler(self.trans[direction]).squeeze().T
        else:
            return grid


# def apply_trans(trans, t):
#     return t @ trans[:3, :3].T + trans[:3, 3]

d = Path("/mnt/scratch/personal/jesperdn/training_data_deformations")
s1 = "sub-0003"
s2 = "sub-0004"

s1_deform = torch.tensor(nib.load(d / f"ABIDE.{s1}.deform.nii").get_fdata()).float()
s1_deform = s1_deform.permute((3, 0, 1, 2))

s2_deform = torch.tensor(nib.load(d / f"ABIDE.{s2}.deform.nii").get_fdata()).float()
s2_deform = s2_deform.permute((3, 0, 1, 2))


A1 = torch.tensor(np.loadtxt(d / f"ABIDE.{s1}.affine.txt")).float()
A2 = torch.tensor(np.loadtxt(d / f"ABIDE.{s2}.affine.txt")).float()
A1B = torch.tensor(np.loadtxt(d / f"ABIDE.{s1}.affine_backward.txt")).float()
A2B = torch.tensor(np.loadtxt(d / f"ABIDE.{s2}.affine_backward.txt")).float()

S1A = torch.tensor(nib.load(f"/mnt/projects/CORTECH/nobackup/training_data/ABIDE.{s1}.T1w.nii").affine).float()
S1AB = torch.linalg.inv(S1A)
S2A = torch.tensor(nib.load(f"/mnt/projects/CORTECH/nobackup/training_data/ABIDE.{s2}.T1w.nii").affine).float()
S2AB = torch.linalg.inv(S2A)

s1_deform_backward = torch.tensor(
    nib.load(d / f"ABIDE.{s1}.deform_backward.nii").get_fdata()
).float()
s1_deform_backward = s1_deform_backward.permute((3, 0, 1, 2))

s2_deform_backward = torch.tensor(
    nib.load(d / f"ABIDE.{s2}.deform_backward.nii").get_fdata()
).float()
s2_deform_backward = s2_deform_backward.permute((3, 0, 1, 2))

s1_t1 = torch.tensor(
    nib.load(f"/mnt/projects/CORTECH/nobackup/training_data/ABIDE.{s1}.T1w.nii")
    .get_fdata()
    .squeeze()[None]
).float()
s1_size = s1_t1.shape[-3:]

s2_t1 = torch.tensor(
    nib.load(f"/mnt/projects/CORTECH/nobackup/training_data/ABIDE.{s2}.T1w.nii")
    .get_fdata()
    .squeeze()[None]
).float()
s2_size = s2_t1.shape[-3:]

self = Sub2SubWarpImage(s1_deform, s2_deform_backward, s1_size)
out1 = self(s1_t1)
affine = nib.load(d / f"ABIDE.{s2}.deform_backward.nii").affine
nib.Nifti1Image(out1[0].numpy(), affine).to_filename(d / "out_s1-to-s2.nii")

self = Sub2SubWarpImage(s2_deform, s1_deform_backward, s2_size)
out2 = self(s2_t1)
affine = nib.load(d / f"ABIDE.{s1}.deform_backward.nii").affine
nib.Nifti1Image(out2[0].numpy(), affine).to_filename(d / "out_s2-to-s1.nii")


size = [i for i in s2_size]

metadata = dict(
    head=[2, 0, 20],
    valid="1  # volume info valid",
    filename="vol.nii",
    volume=size,
    voxelsize=[1, 1, 1],
    xras=[-1, 0, 0],
    yras=[0, 0, -1],
    zras=[0, 1, 0],
    cras=[0, 0, 0],
)

s1_white = torch.load(f"/mnt/projects/CORTECH/nobackup/training_data/ABIDE.{s1}.surf_dir/lh.white.5.target.pt")
s1_pial = torch.load(f"/mnt/projects/CORTECH/nobackup/training_data/ABIDE.{s1}.surf_dir/lh.pial.5.target.pt")

self = Sub2SubWarpSurface(s1_deform_backward, s2_deform, s1_size)

out1 = self(s1_white)
nib.freesurfer.write_geometry(
    d / "surf_lh.white_s1-to-s2",
    (out1.float() @ S2A[:3,:3].T + S2A[:3, 3]).numpy(), top[5].faces.numpy(), volume_info=metadata)

out1 = self(s1_pial)
nib.freesurfer.write_geometry(
    d / "surf_lh.pial_s1-to-s2",
    (out1.float() @ S2A[:3,:3].T + S2A[:3, 3]).numpy(), top[5].faces.numpy(), volume_info=metadata)


s1_white = torch.load(f"/mnt/projects/CORTECH/nobackup/training_data/ABIDE.{s1}.surf_dir/rh.white.5.target.pt")
s1_pial = torch.load(f"/mnt/projects/CORTECH/nobackup/training_data/ABIDE.{s1}.surf_dir/rh.pial.5.target.pt")

out1 = self(s1_white)
nib.freesurfer.write_geometry(
    d / f"surf_rh.white_s1-to-s2",
    (out1.float() @ S2A[:3,:3].T + S2A[:3, 3]).numpy(), top[5].faces.numpy(), volume_info=metadata)

out1 = self(s1_pial)
nib.freesurfer.write_geometry(
    d / f"surf_rh.pial_s1-to-s2",
    (out1.float() @ S2A[:3,:3].T + S2A[:3, 3]).numpy(), top[5].faces.numpy(), volume_info=metadata)


"""
# AFFINE TRANSFORM

A1 = torch.tensor(np.loadtxt(d / f"ABIDE.{s1}.affine.txt")).float()
A2 = torch.tensor(np.loadtxt(d / f"ABIDE.{s2}.affine.txt")).float()
A1B = torch.tensor(np.loadtxt(d / f"ABIDE.{s1}.affine_backward.txt")).float()
A2B = torch.tensor(np.loadtxt(d / f"ABIDE.{s2}.affine_backward.txt")).float()

S1A = torch.tensor(nib.load(f"/mnt/projects/CORTECH/nobackup/training_data/ABIDE.{s1}.T1w.nii").affine).float()
S1AB = torch.linalg.inv(S1A)
S2A = torch.tensor(nib.load(f"/mnt/projects/CORTECH/nobackup/training_data/ABIDE.{s2}.T1w.nii").affine).float()
S2AB = torch.linalg.inv(S2A)


grid = torch.stack(torch.meshgrid([torch.arange(i) for i in s2_size], indexing="ij")).float()
grid = grid.permute((1,2,3,0))
Ax = S1AB @ A1 @ A2B @ S2A
grid = grid @ Ax[:3,:3].T + Ax[:3,3]

sampler = GridSample(grid, s1_size)
out11 = sampler(s1_t1)
nib.Nifti1Image(out11[0].numpy(), S2A).to_filename(d / f"affine_{s1}-to-{s2}.nii")


grid = torch.stack(torch.meshgrid([torch.arange(i) for i in s1_size], indexing="ij")).float()
grid = grid.permute((1,2,3,0))
Ay = S2AB @ A2 @ A1B @ S1A
grid = grid @ Ay[:3,:3].T + Ay[:3,3]

sampler = GridSample(grid, s2_size)
out11 = sampler(s2_t1)
nib.Nifti1Image(out11[0].numpy(), S2A).to_filename(d / f"affine_{s2}-to-{s1}.nii")

"""


class Sub2SubWarpImage(BaseTransform):
    def __init__(
        self,
        s1_deform: torch.Tensor,
        s2_deform_backward: torch.Tensor,
        s1_size: torch.Tensor | torch.Size,
        device: None | torch.device = None,
    ) -> None:
        """Warp an image from subject 1 (voxel) space to subject 2 (voxel)
        space. The mapping is achieved by composing the supplied deformation
        fields, i.e.,

                      s1_deform            s2_deform_backward
            S1[vox]     <---     MNI[vox]         <---          S2[vox]

        such that, when applied to an image in subject 1 space, they will pull
        the values into the space of subject 2.

        Parameters
        ----------
        s1_deform: torch.Tensor
            Forward deformation field of subject 1. The forward deformation
            maps each voxel in MNI152 space to the corresponding coordinate in
            subject 1 *voxel* space. Forward deformations has MNI affine and
            MNI size.
        s2_deform_backward: torch.Tensor
            Backward deformation field of subject 2. The backward deformation
            maps each voxel in subject 2 space to the corresponding coordinate
            in MNI *voxel* space. Backward deformations has subject affine and
            subject size.
        s1_size: torch.Tensor | torch.Size
            The spatial dimensions of subject 1 image space.
        """
        # images are (C,W,H,D) !!

        super().__init__(device)
        mni_image_shape = s1_deform.shape[-3:]

        grid = s2_deform_backward.permute((1, 2, 3, 0))     # MNI voxel coordinates
        grid = GridSample(grid, mni_image_shape)(s1_deform) # S1 voxel coordinates
        grid = grid.permute((1, 2, 3, 0))
        # The grid now contains the voxel coordinates in S1 corresponding to
        # each voxel in S2, i.e., we have a S1[vox] to S2[vox] mapping.
        self.image_sampler = GridSample(grid, s1_size)

    def forward(self, image):
        return self.image_sampler(image)


class Sub2SubWarpSurface(BaseTransform):
    def __init__(
        self,
        s1_deform_backward: torch.Tensor,
        s2_deform: torch.Tensor,
        s2_size: torch.Tensor | torch.Size,
        device: None | torch.device = None,
    ) -> None:
        """Warp a surface from subject 1 (voxel) space to subject 2 (voxel)
        space. The mapping is achieved by composing the supplied deformation
        fields, i.e.,

                      s2_deform            s1_deform_backward
            S2[vox]     <---     MNI[vox]         <---          S1[vox]

        such that, when applied to a surface in subject 1 space, they will push
        the vertices into the space of subject 2. Note that this is the inverse
        operation of `Sub2SubWarpImage`: Sub2SubWarpImage deforms the image
        grid of S2 to S1 space and pulls the values. Sub2SubWarpSurface pushes
        coordinates in S1 space to S2 space.

        Parameters
        ----------
        s1_deform_backward :
            Backward deformation field of subject 1. The backward deformation
            maps each voxel in subject 1 space to the corresponding coordinate
            in MNI *voxel* space. Backward deformations has subject affine and
            subject size.
        s2_deform :
            Forward deformation field of subject 2. The forward deformation
            maps each voxel in MNI152 space to the corresponding coordinate in
            subject 2 *voxel* space. Forward deformations has MNI affine and
            MNI size.
        """
        super().__init__(device)
        mni_image_shape = s2_deform.shape[-3:]

        grid = s1_deform_backward.permute((1, 2, 3, 0))     # MNI voxel coordinates
        sampler = GridSample(grid, mni_image_shape)
        # The field now contains the voxel coordinates in S2 space corresponding
        # to each voxel in S1, i.e., we have a S2[vox] to S1[vox] mapping.
        self.deformation_field = sampler(s2_deform) # S2 voxel coordinates

    def forward(self, surface):
        sampler = GridSample(surface, s2_size)
        return sampler(self.deformation_field).squeeze().T


class CheckCoordsInside(BaseTransform):
    def __init__(
        self, size, raise_on_outsize: bool = False, device: None | torch.device = None
    ) -> None:
        super().__init__(device)
        self.size = torch.as_tensor(size, device=self.device)
        self.raise_on_outsize = raise_on_outsize
        self.msg_header = "Cortical surface is partly outside of FOV."

    def forward(self, coords):
        msg = [self.msg_header]
        outside = False
        if torch.any((smin := coords.amin(0)) < 0):
            msg.append(f"  {torch.round(smin, decimals=2)} < (0, 0, 0)")
            outside = True
        if torch.any((smax := coords.amax(0)) > self.size):
            msg.append(f"  {torch.round(smax, decimals=2)} > {self.size}")
            outside = True

        if outside:
            msg = "\n".join(msg)
            if self.raise_on_outsize:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        return coords


# class SurfaceDeformation(BaseTransform):
#     def __init__(
#         self,
#         linear_trans: RandLinearTransform,
#         nonlinear_trans: RandNonlinearTransform,
#         # nonlinear_field: ,
#         # out_size: ,
#         out_center: torch.Tensor,
#         device: None | torch.device = None,
#     ) -> None:
#         super().__init__(device)
#         self.linear_trans = linear_trans
#         self.nonlinear_trans = nonlinear_trans
#         self.out_size = self.nonlinear_trans.out_size
#         # the center of the final FOV
#         self.out_center = out_center
#         # center in the output image
#         # self.center_in_out_image = GridCentering(self.out_size, device=self.device)

#     def deform_surface(self, s):
#         # surface_center = 0.5 * (s.amax(0) - s.amin(0))

#         # center around (0,0,0)
#         # s -= surface_center
#         s -= self.out_center

#         s = self.linear_trans(s, inverse=True)

#         # place appropriately in output FOV
#         # s = self.center_in_out_image(s + self.out_center + surface_center)
#         s = s + center_from_shape(self.out_size)

#         if self.nonlinear_trans.should_apply_transform():
#             sample = GridSample(s, self.nonlinear_trans.out_size)
#             s += sample(self.nonlinear_trans.trans["backward"])

#         if torch.any((smin := s.amin(0)) < 0) or torch.any(
#             (smax := s.amax(0)) > self.out_size
#         ):
#             smax = s.amax(0) if "smax" not in vars() else smax
#             warnings.warn(
#                 (
#                     "Cortical surface is partly outside of FOV.\n"
#                     f"  {tuple(i.item() for i in smin)} > (0,0,0)"
#                     f"  {tuple(i.item() for i in smax)} < {self.out_size}."
#                 )
#             )

#         return s

#     def forward(self, surface: dict | torch.Tensor):
#         if isinstance(surface, torch.Tensor):
#             return self.deform_surface(surface)
#         else:
#             return {k: self.forward(v) for k, v in surface.items()}


class Resize(BaseTransform):
    def __init__(self, size: torch.Tensor | tuple | list, mode="trilinear"):
        super().__init__()
        self.size = tuple(size)
        self.mode = mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image[None]
        if image.ndim == 4:
            image = image[None]

        return torch.nn.functional.interpolate(
            input=image,
            size=self.size,
            mode=self.mode,
            align_corners=True,
        ).squeeze(0)


# class RandFlipImage(RandomizableTransform):
#     def __init__(self, prob: float = 1):
#         super().__init__(prob)

#     def forward(self):
#         raise NotImplementedError


# class RandLRFlipSurface(RandomizableTransform):
#     def __init__(self, size, prob: float = 1.0):
#         super().__init__(prob)
#         self.flip_hemi = dict(zip(constants.HEMISPHERES, constants.HEMISPHERES[::-1]))
#         self.dim = 0
#         self.width = size[self.dim] # assume RAS : [W,H,D]

#     def _flip_surface_coords(self, s):
#         s[:, self.dim] *= -1.0
#         s[:, self.dim] += self.width - 1.0

#     def forward(self, surfaces):
#         if not self.should_apply_transform():
#             return surfaces

#         for hemi, surfs in surfaces.items():
#             if isinstance(surfs, dict):
#                 for s in surfs:
#                     self._flip_surface_coords(surfaces[hemi][s])
#             else:
#                 self._flip_surface_coords(surfs)

#         # NOTE
#         # swap label (e.g., lh -> rh) when flipping image. Not sure if
#         # this is the best thing to do but at least ensures some kind
#         # of consistency: when flipping then faces need to be reordered
#         # which is also the case for rh compared to lh.
#         return {self.flip_hemi[h]: v for h, v in surfaces.items()}


class SpatialSize(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: torch.Tensor):
        return torch.as_tensor(image.shape[-3:], device=image.device)


class Grid(BaseTransform):
    def __init__(self, size, device: None | torch.device = None) -> None:
        super().__init__(device)
        self.size = size

    def forward(self):
        return torch.stack(
            torch.meshgrid(
                [
                    torch.arange(s, dtype=torch.float, device=self.device)
                    for s in self.size
                ],
                indexing="ij",
            ),
            dim=-1,
        )


class GridCentering(BaseTransform):
    def __init__(
        self, size, align_corners: bool = True, device: None | torch.device = None
    ) -> None:
        super().__init__(device)
        self.align_corners = align_corners
        self.size = torch.as_tensor(size, device=self.device)
        if self.align_corners:
            self.center = 0.5 * (self.size - 1.0)
        else:
            self.center = 0.5 * self.size

    def forward(self, grid: torch.Tensor):
        return grid - self.center


class GridNormalization(GridCentering):
    def __init__(
        self, size, align_corners: bool = True, device: None | torch.device = None
    ) -> None:
        super().__init__(size, align_corners, device)

    def forward(self, grid: torch.Tensor):
        return super().forward(grid) / self.center


class GridSample(BaseTransform):
    def __init__(
        self,
        grid: torch.Tensor,
        size: None | torch.Tensor | torch.Size = None,
        assume_normalized: bool = False,
        # device: None | torch.device = None,
    ):
        """_summary_

        Parameters
        ----------
        grid : torch.Tensor
            Grid of shape (V, 3) or (W, H, D, 3).
        size : None | torch.Tensor, optional
           Size of the image to be interpolated from (i.e., the size of the
           image used when calling GridSample). The grid is normalized by this
           shape.
        assume_normalized : bool, optional
            Assume that `grid` is already normalized in which case `size` is
            not required (default = False).
        """
        super().__init__()

        self.grid = grid
        if not assume_normalized:
            assert (
                size is not None
            ), "`size` must be provided when `assume_normalized = False`"
            assert len(size) == 3
        self.size = size
        self.assume_normalized = assume_normalized

        self.grid_ndim = len(self.grid.shape)
        assert self.grid_ndim in {2, 4}

        # normalized grid
        if self.assume_normalized:
            self.norm_grid = grid
        else:
            self.norm_grid = GridNormalization(self.size, device=grid.device)(grid)

        if self.grid_ndim == 2:
            self.norm_grid = self.norm_grid[None, :, None, None]  # 1,W,1,1,3
        elif self.grid_ndim == 4:
            self.norm_grid = self.norm_grid[None]  # 1,W,H,D,3

    def forward(self, image, padding_mode="border"):
        """_summary_

        Parameters
        ----------
        image : _type_
            Image to interpolate from. The dimensions are (C, H, W, D) where
            (H, W, D) should match the spatial size provided at initialization
            (if any).

        Returns
        -------
        _type_
            _description_
        """
        # image : (C, W, H, D)
        # grid  : (V, 3) or (WW, HH, DD, 3)
        if self.grid is None:
            return image

        if not self.assume_normalized:
            assert (image_size := tuple(image.size()[-3:])) == tuple(
                self.size
            ), f"Expected image with spatial dimensions {self.size} but got {image_size}."

        image_ndim = len(image.shape)
        assert image_ndim in {3, 4}

        # image
        # [n,c,]x,y,z -> n,c,z,y,x (!)
        if image_ndim == 3:
            image = image[None, None]
        elif image_ndim == 4:
            image = image[None]
        image = image.transpose(2, 4)  # NOTE xyz -> zyx (!)

        # NOTE `sample` has original xyz ordering! (N,C,W,H,D)
        if image.is_floating_point():
            sample = torch.nn.functional.grid_sample(
                image,
                self.norm_grid,
                align_corners=True,
                mode="bilinear",
                padding_mode=padding_mode,
            )
        else:
            # Cast to float and back since grid_sample doesn't support int (
            # at least not on cuda)
            sample = torch.nn.functional.grid_sample(
                image.float(),
                self.norm_grid,
                align_corners=True,
                mode="nearest",
                padding_mode=padding_mode,
            ).to(image.dtype)

        if self.grid_ndim == 2:
            sample.squeeze_((2, 3))
        sample.squeeze_(0)

        return sample


class SurfaceBoundingBox(BaseTransform):
    def __init__(self, device: None | torch.device = None) -> None:
        super().__init__(device)

    @staticmethod
    def bbox(x):
        """Bounding box of 2D array with (points, coordinates)."""
        return torch.stack((x.amin(0), x.amax(0)))

    def forward(self, surface: dict[str, torch.Tensor]):
        return {h: self.bbox(s) for h, s in surface.items()}


class CenterFromString(BaseTransform):
    def __init__(
        self,
        size: torch.Tensor,
        surface_bbox: dict[str, torch.Tensor] | None = None,
        align_corners: bool = True,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        self.size = size
        self.surface_bbox = surface_bbox
        self.align_corners = align_corners
        if self.surface_bbox is None:
            self.valid_centers = {"image", "random"}
        else:
            self.valid_centers = None

    def forward(self, center):
        if self.valid_centers is not None:
            assert center in self.valid_centers

        match center:
            case "brain":
                raise NotImplementedError
                # center =
            case "image":
                res = 0.5 * (self.size - 1.0)
            case "lh":
                res = self.surface_bbox["lh"].mean(0)
            case "random":
                raise NotImplementedError
                # center =
            case "rh":
                res = self.surface_bbox["rh"].mean(0)

        # When align_corners = True (in grid sampling), this, together with the
        # assertion that the FOV size is divisible by 2, ensures that we sample
        # exactly in the center of a voxel when there is no linear or nonlinear
        # deformation (we could get almost identical results with cropping).
        # This is important, especially for label images, as otherwise we are
        # always sampling exactly in between two voxels which will result in
        # label images that look downsampled. (Also, the exact center is not
        # important.)
        return res.floor() + 0.5 if self.align_corners else res


class RandResolution(RandomizableTransform):
    """Simulate a"""

    def __init__(
        self,
        out_size,
        in_res,
        res_sampler: str = "ResolutionSamplerDefault",
        photo_mode: bool = False,
        photo_spacing: float | None = None,
        photo_thickness: float = 0.001,
        photo_res_sampler: str = "ResolutionSamplerPhoto",
        prob: float = 1.0,
        device: None | torch.device = None,
    ):
        super().__init__(prob, device)

        self.in_res = in_res
        self.out_size = torch.as_tensor(out_size, device=self.device)
        self.photo_mode = photo_mode
        self.photo_spacing = photo_spacing
        self.resize = Resize(out_size)

        # Apply resolution and slice thickness blurring if different from the
        # target resolution
        if self.should_apply_transform():
            if self.photo_mode:
                out_res, thickness = getattr(resolution_sampler, photo_res_sampler)(
                    self.in_res,
                    photo_spacing,
                    photo_thickness,
                    self.device,
                )
            else:
                out_res, thickness = getattr(resolution_sampler, res_sampler)(
                    self.device
                )()

            sigma = (
                (0.85 + 0.3 * torch.rand(1, device=self.device))
                * torch.log(torch.tensor([5.0], device=self.device))
                / torch.pi
                * thickness
                / self.in_res
            )
            # no blur if thickness is equal to the resolution of the training
            # data
            sigma[thickness <= self.in_res] = 0.0
            self.out_res = out_res
            self.sigma = tuple(s.item() for s in sigma)
            self.gaussian_blur = GaussianSmooth(self.sigma, device=self.device)
        else:
            self.out_res = in_res
            self.sigma = None
            self.gaussian_blur = None

    def forward(self, image):
        if not self.should_apply_transform():
            return image

        # sanity check
        # assert tuple(self.out_size) == tuple(image.size()[-3:])

        ds_size = self.out_size * self.in_res / self.out_res
        ds_size = ds_size.to(torch.int)

        factors = ds_size / self.out_size
        delta = (1.0 - factors) / (2.0 * factors)
        hv = tuple(
            torch.arange(d, d + ns / f, 1 / f, device=self.device)[:ns]
            for d, ns, f in zip(delta, ds_size, factors)
        )
        small_grid = torch.stack(torch.meshgrid(*hv, indexing="ij"), dim=-1)

        # blur and downsample
        # image = monai.transforms.GaussianSmooth(self.sigma)(image)
        image = self.gaussian_blur(image)
        image = GridSample(small_grid, self.out_size)(image)

        # is this necessary?
        # SYN_noisy = self.eliminate_negative_values(SYN_noisy)
        # SYN_final = self.normalize_intensity(SYN_resized)

        return self.resize(image)
