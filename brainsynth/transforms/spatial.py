import warnings

import torch

from brainsynth.transforms.base import BaseTransform, RandomizableTransform


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
        self, translation: torch.Tensor, device: None | torch.device = None
    ) -> None:
        super().__init__(device)
        self.translation = translation.to(self.device)

    def forward(self, x: torch.Tensor):
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
        integrate_svf: bool = False,
        n_steps: int = 8,
        photo_mode: bool = False,
        device: None | torch.device = None,
        prob: float = 1.0,
    ):
        """
        To be consistent with torch convention that the channel dimension is
        before the spatial dimensions, we create the deformation field such
        that the deformations are [C,W,H,D] although when applying it in
        `compute_deformed_grid` we use it as [W,H,D,C].


        n_steps
            Number of integration steps for

        """
        super().__init__(prob, device)

        self.out_size = out_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.std_max = std_max
        self.integrate_svf = integrate_svf
        self.n_steps = n_steps

        if not self.should_apply_transform():
            self.trans = dict(forward=None, backward=None)
            return

        nonlin_scale = scale_min + torch.rand(1) * (scale_max - scale_min)
        size_F_small = torch.round(nonlin_scale * out_size).to(torch.int)

        if photo_mode:
            size_F_small[1] = torch.round(out_size[1] / self.spacing).to(
                size_F_small.dtype
            )
        nonlin_std = std_max * torch.rand(1)

        # Channel first (as is normal)
        ft = nonlin_std * torch.randn([3, *size_F_small], device=self.device)
        ft = Resize(out_size)(ft)

        if photo_mode:
            # no deformation in AP direction
            ft[1] = 0

        ft = self.channel_last(ft)

        if integrate_svf:
            # this is slow on the CPU
            steplength = 1.0 / (2.0**n_steps)

            # forward integration
            Fsvf = ft * steplength
            for _ in torch.arange(n_steps):
                # Fsvf += self.apply_grid_sample(
                #     Fsvf, grid + self.as_channel_last(Fsvf)
                # )
                resample = GridSample(grid + self.channel_last(Fsvf), out_size)
                Fsvf += resample(Fsvf)

            # backward integration
            Fsvf_neg = -ft * steplength
            for _ in torch.arange(self.n_steps):
                # Fsvf_neg += self.apply_grid_sample(
                #     Fsvf_neg, grid + self.as_channel_last(Fsvf_neg)
                # )
                resample = GridSample(grid + self.channel_last(Fsvf_neg), out_size)
                Fsvf_neg += resample(Fsvf_neg)

            ft = Fsvf
            bt = Fsvf_neg
        else:
            bt = None

        self.trans = dict(forward=ft, backward=bt)

    @staticmethod
    def channel_last(x):
        return x.permute((3, 1, 2, 0))

    def forward(self, grid, direction="forward"):
        if self.should_apply_transform():
            return grid + self.channel_last(self.trans[direction])
        else:
            return grid


class SurfaceDeformation(BaseTransform):
    def __init__(
        self, linear_trans, nonlinear_trans, device: None | torch.device = None
    ) -> None:
        super().__init__(device)
        self.linear_trans = linear_trans
        self.nonlinear_trans = nonlinear_trans

    def deform_surface(self, s):
        surface_center = 0.5 * (s.amax(1) - s.amin(1))
        image_center = center_from_shape(self.nonlinear_trans.out_size, align_corners=True)

        # center around (0,0,0)
        s -= surface_center  # s -= self.translation

        s = self.linear_trans(s, inverse=True)

        # center in output FOV
        s += image_center  # s += self.center

        if self.nonlinear_trans.should_apply_transform():
            sample = GridSample(s, self.nonlinear_trans.out_size)
            s += sample(self.nonlinear_trans.trans["backward"])

        if (smin := s.amin() < 0) or torch.any(smax := s.amax(0) > self.fov_size):
            warnings.warn(
                (
                    "Cortical surface is partly outside of FOV. BBOX of FOV is "
                    f"{(0,0,0), self.fov_size}. BBOX of surface is "
                    f"{smin(0), smax(0)}"
                )
            )

        return s

    def forward(self, surface: dict | torch.Tensor):
        if isinstance(surface, torch.Tensor):
            return self.deform_surface(surface)
        else:
            return {k: self.forward(v) for k, v in surface.items()}


def center_from_shape(size, align_corners=True):
    if align_corners:
        return 0.5 * (size - 1.0)
    else:
        return 0.5 * size


class Resize(BaseTransform):
    def __init__(self, size: torch.Tensor | tuple | list, mode="bilinear"):
        super().__init__()
        self.size = tuple(size)
        self.mode = mode or "bilinear"  # trilinear when `image` is 5-D

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
        size: None | torch.Tensor = None,
        assume_normalized: bool = False,
        # device: None | torch.device = None,
    ):
        super().__init__()

        self.grid = grid
        if not assume_normalized:
            assert (
                size is not None
            ), "`size` must be provided when `assume_normalized = False`"
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

    def forward(self, image, **kwargs):
        # image : (C, W, H, D)
        # grid  : (V, 3) or (WW, HH, DD, 3)
        if self.grid is None:
            return image

        if not self.assume_normalized:
            assert (
                (image_size := tuple(image.size()[-3:])) == tuple(self.size)
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
                image, self.norm_grid, align_corners=True, mode="bilinear", **kwargs
            )
        else:
            # Cast to float and back since grid_sample doesn't support int (
            # at least not on cuda)
            sample = torch.nn.functional.grid_sample(
                image.float(), self.norm_grid, align_corners=True, mode="nearest", **kwargs
            ).to(image.dtype)

        if self.grid_ndim == 2:
            sample.squeeze_((2, 3))
        sample.squeeze_(0)

        return sample


class SurfaceBoundingBox(BaseTransform):
    def __init__(self, device: None | torch.device = None) -> None:
        super().__init__(device)

    def forward(self, surface: dict[str, torch.Tensor]):
        return {h:torch.stack((s.amin(0), s.amax(0))) for h,s in surface.items()}


class CenterFromString(BaseTransform):
    def __init__(
        self,
        size: torch.Tensor,
        surface_bbox: dict[str, torch.Tensor] | None = None,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        self.size = size
        self.surface_bbox = surface_bbox
        if self.surface_bbox is None:
            self.valid_centers = {"image", "random"}
        else:
            self.valid_centers = None

    def forward(self, center):
        if self.valid_centers is not None:
            assert center in self.valid_centers
        match center:
            case "brain":
                return
            case "image":
                return 0.5 * (self.size - 1.0)
            case "lh":
                return self.surface_bbox["lh"].mean(0)
            case "random":
                return
            case "rh":
                return self.surface_bbox["rh"].mean(0)
            # case torch.tensor:
            #     return center
            # case list | tuple:
            #     return torch.tensor(center, device=self.device)


class RandResolution(RandomizableTransform):
    """Simulate a"""

    def __init__(
        self,
        out_size,
        in_res,
        photo_mode: bool = False,
        photo_spacing: float | None = None,
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
                out_res, thickness = self.resolution_photo_mode()
            else:
                out_res, thickness = self.resolution_sampler()
            stds = (
                (0.85 + 0.3 * torch.rand(1, device=self.device))
                * torch.log(torch.tensor([5.0], device=self.device))
                / torch.pi
                * thickness
                / self.in_res
            )
            # no blur if thickness is equal to the resolution of the training
            # data
            stds[thickness <= self.in_res] = 0.0
            self.out_res = out_res
            self.stds = stds
        else:
            self.out_res = in_res
            self.stds = None

    def resolution_photo_mode(self, slice_thickness=0.001):
        out_res = torch.tensor(
            [self.in_res[0], self.spacing, self.in_res[2]], device=self.device
        )
        thickness = torch.tensor(
            [self.in_res[0], slice_thickness, self.in_res[2]], device=self.device
        )
        return out_res, thickness

    def resolution_sampler(self):

        # if self.do_task["photo_mode"]:

        # else:
        #     out_res = in_res.clone()
        #     thickness = in_res.clone()
        #     # out_res, thickness = resolution_sampler(self.device)
        # return out_res, thickness

        with torch.device(self.device):
            r = torch.rand(1)
            # if r < 0.25:  # 1mm isotropic
            #     resolution = torch.tensor([1.0, 1.0, 1.0])
            #     thickness = torch.tensor([1.0, 1.0, 1.0])
            if r < 0.33:
                # clinical (low-res in one dimension)
                resolution = torch.tensor([1.0, 1.0, 1.0])
                thickness = torch.tensor([1.0, 1.0, 1.0])
                idx = torch.randint(0, 3, (1,))
                resolution[idx] = 2.5 + 2 * torch.rand(1)
                thickness[idx] = torch.minimum(
                    resolution[idx], 4.0 + 2.0 * torch.rand(1)
                )
            elif r < 0.67:
                # low-field: stock sequences (always axial)
                resolution = torch.tensor([1.3, 1.3, 3.0]) + 0.4 * torch.rand(3)
                thickness = resolution.clone()
            else:
                # low-field: isotropic-ish (also good for scouts)
                resolution = 2.0 + 1.0 * torch.rand(3)
                thickness = resolution.clone()

        return resolution, thickness

    def forward(self, image):
        if self.stds is None:
            return image

        # sanity check
        # assert tuple(self.out_size) == tuple(image.size()[-3:])

        ds_size = self.out_size * self.in_res / self.out_res
        ds_size = ds_size.to(torch.int)

        factors = ds_size / self.out_size
        delta = (1.0 - factors) / (2.0 * factors)
        hv = tuple(
            torch.arange(d, d + ns / f, 1 / f)[:ns]
            for d, ns, f in zip(delta, ds_size, factors)
        )
        small_grid = torch.stack(torch.meshgrid(*hv, indexing="ij"), dim=-1)

        # blur and downsample
        image = monai.transforms.GaussianSmooth(self.stds)(image)
        image = GridSample(small_grid, self.out_size)(image)

        # is this necessary?
        # SYN_noisy = self.eliminate_negative_values(SYN_noisy)
        # SYN_final = self.normalize_intensity(SYN_resized)

        return self.resize(image)
