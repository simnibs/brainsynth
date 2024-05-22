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

    def forward(self, grid):
        """`grid` has coordinates in the last dimension!"""
        if self.should_apply_transform():
            return grid @ self.trans.T
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

        if not self.should_apply_transform():
            self.trans = dict(forward=None, backward=None)
            return

        self.out_size = out_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.std_max = std_max
        self.integrate_svf = integrate_svf
        self.n_steps = n_steps

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
            nonlin_deform = self.channel_last(self.forward_trans)
            return grid + self.forward_trans
        else:
            return grid


class Resize(BaseTransform):
    def __init__(self, size: torch.Tensor | tuple | list, mode="bilinear"):
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


class GridSample(BaseTransform):
    def __init__(self, grid: torch.Tensor, out_size: torch.Tensor):
        self.grid = grid
        self.grid_ndim = len(self.grid.shape)
        assert self.grid_ndim in {2, 4}

        self.out_size = out_size

        # normalized grid
        self.norm_grid = self.normalize_coordinates()
        if self.grid_ndim == 2:
            self.norm_grid = self.norm_grid[None, :, None, None]  # 1,W,1,1,3
        elif self.grid_ndim == 4:
            self.norm_grid = self.norm_grid[None]  # 1,W,H,D,3

    def center_from_shape(self):
        return 0.5 * (self.out_size - 1.0)  # -1.0 because of align_corners=True

    def normalize_coordinates(self):
        c = self.center_from_shape()
        return (self.grid - c) / c

    def forward(self, image, **kwargs):
        # image : (C, W, H, D)
        # grid  : (V, 3) or (W, H, D, 3)
        if self.grid is None:
            return image

        image_ndim = len(image.shape)
        assert image_ndim in {3, 4}

        assert (
            self.out_size == image.shape[-3:]
        ), f"Wrong image size, expected {self.out_size} but got {image.shape[-3:]}"
        # shape = torch.tensor(image.shape[-3:])

        # image
        # [n,c,]x,y,z -> n,c,z,y,x (!)
        if image_ndim == 3:
            image = image[None, None]
        elif image_ndim == 4:
            image = image[None]
        image = image.transpose(2, 4)  # NOTE xyz -> zyx (!)

        # NOTE sample has original xyz ordering! (N,C,W,H,D)
        sample = torch.nn.functional.grid_sample(
            image, self.norm_grid, align_corners=True, **kwargs
        )
        if self.grid_ndim == 2:
            sample.squeeze_((2, 3))
        sample.squeeze_(0)

        return sample


class RandResolution(RandomizableTransform):
    """Simulate a"""

    def __init__(self, out_size, in_res, out_res, prob: float = 1.0):
        super().__init__(prob)

        self.in_res = in_res
        self.out_res = out_res
        self.resize = Resize(out_size)
        self.device = self.in_res.device

        # Apply resolution and slice thickness blurring if different from the
        # target resolution
        if not self.in_res.equal(self.out_res):
            super().randomize()
            if self.should_apply_transform():
                out_res, thickness = self.resolution_sampler()

                stds = (
                    (0.85 + 0.3 * torch.rand([1], device=self.device))
                    * torch.log(torch.tensor([5.0], device=self.device))
                    / torch.pi
                    * thickness
                    / self.in_res
                )
                # no blur if thickness is equal to the resolution of the training
                # data
                stds[thickness <= self.in_res] = 0.0
                self.stds = stds
            else:
                self.stds = None
        else:
            self.stds = None

    def resolution_sampler(self):

        # if self.do_task["photo_mode"]:
        #     out_res = torch.tensor([in_res[0], self.spacing, in_res[2]])
        #     thickness = torch.tensor([in_res[0], 0.0001, in_res[2]])
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
