import torch

from brainsynth.transforms.base import RandomizableTransform

class SpatialCrop(torch.nn.Module):
    def __init__(
            self,
            in_size: torch.Tensor,
            out_size: torch.Tensor,
            out_center: torch.Tensor,
        ) -> None:
        self.in_size = in_size
        self.out_size = out_size
        self.out_center = out_center

        halfsize = 0.5 * (out_size - 1)
        start = torch.ceil(out_center - halfsize).int()
        stop = torch.floor(out_center + halfsize).int()
        stop += out_size - (stop-start)
        slices = tuple((a.item(), b.item()) for a,b in zip(start, stop))

        # sample these slices in the original image
        self.slice_in = tuple(
            slice(max(slc[0], 0), min(slc[1], ins.item())) for slc,ins in zip(slices, in_size)
        )
        # place them in these positions in the new image
        self.slice_out = tuple(
            slice(0 - min(0, slc[0]), outs.item() - max(0, slc[1]-ins.item())) for slc,ins,outs in zip(slices, in_size, out_size)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out_size = torch.zeros(image.ndim, dtype=self.out_size.dtype)
        out_size[-3:] = self.out_size
        for i,s in enumerate(image.shape[:-3]):
            out_size[i] = s

        out = torch.zeros(tuple(out_size), dtype=image.dtype)
        out[..., *self.slice_out] = image[..., *self.slice_in]
        return out


class RandLinearTransform(RandomizableTransform):
    def __init__(self, max_scale, max_rotation, max_shear, prob: float = 1):
        super().__init__(prob)

        self.max_scale = max_scale
        self.max_rotation = max_rotation
        self.max_shear = max_shear

        if not self._do_transform:
            self.linear_transform = None
            self.scale_distance = 1.0
            return

        rotations = (
            (2 * self.max_rotation * torch.rand(3) - self.max_rotation) / 180.0 * torch.pi
        )
        shears = 2 * self.max_shear * torch.rand(3) - self.max_shear
        scalings = 1 + (2 * self.max_scale * torch.rand(3) - self.max_scale)
        # we divide distance maps by this, not perfect, but better than nothing
        scaling_factor_distances = torch.prod(scalings) ** 0.33333333333

        self.linear_transform = self.generate_linear_transform(
            rotations, shears, scalings, self.device
        )
        self.scale_distance = scaling_factor_distances

    @staticmethod
    def generate_linear_transform(rot, shear, scale, device):
        with torch.device(device):
            cosR = torch.cos(rot)
            sinR = torch.sin(rot)

            Rx = torch.tensor(
                [
                    [1,       0,        0],
                    [0, cosR[0], -sinR[0]],
                    [0, sinR[0],  cosR[0]],
                ]
            )
            Ry = torch.tensor(
                [
                    [ cosR[1], 0, sinR[1]],
                    [       0, 1,       0],
                    [-sinR[1], 0, cosR[1]],
                ]
            )
            Rz = torch.tensor(
                [
                    [cosR[2], -sinR[2], 0],
                    [sinR[2],  cosR[2], 0],
                    [      0,        0, 1],
                ]
            )

            SHx = torch.tensor([[1, 0, 0], [shear[1], 1, 0], [shear[2], 0, 1]])
            SHy = torch.tensor([[1, shear[0], 0], [0, 1, 0], [0, shear[2], 1]])
            SHz = torch.tensor([[1, 0, shear[0]], [0, 1, shear[1]], [0, 0, 1]])

            A = SHx @ SHy @ SHz @ Rx @ Ry @ Rz

            A[0] *= scale[0]
            A[1] *= scale[1]
            A[2] *= scale[2]

        return A


    def forward(self, grid):
        """`grid` has coordinates in the last dimension!"""
        if self.linear_transform is None:
            return grid
        else:
            return grid @ self.linear_transform.T

class RandNonlinearTransform(RandomizableTransform):
    def __init__(self, size, scale_min, scale_max, std_min, std_max, n_steps, prob: float = 1):
        """
        To be consistent with torch convention that the channel dimension is
        before the spatial dimensions, we create the deformation field such
        that the deformations are [C,W,H,D] although when applying it in
        `compute_deformed_grid` we use it as [W,H,D,C].


        n_steps
            Number of integration steps for

        """
        super().__init__(prob)

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.std_min = std_min
        self.std_max = std_max

        self.n_steps = n_steps

        self.resize = Resize(size)

        if not self.do_task["nonlinear_deform"]:
            self.nonlinear_transform_fwd = None
            self.nonlinear_transform_bwd = None
            return

        nonlin_scale = self.scale_min + torch.rand(1) * (self.scale_max - self.scale_min)
        size_F_small = torch.round(nonlin_scale * self.size).to(torch.int)

        if self.do_task["photo_mode"]:
            size_F_small[1] = torch.round(self.size[1] / self.spacing).to(
                size_F_small.dtype
            )
        nonlin_std = self.std_max * torch.rand(1)

        Fsmall = nonlin_std * torch.randn([3, *size_F_small])

        F = self.resize(Fsmall)

        if self.do_task["photo_mode"]:
            F[1] = 0

        if self.has_surfaces:  # this is slow
            steplength = 1.0 / (2.0**self.n_steps)
            Fsvf = F * steplength
            for _ in torch.arange(self.n_steps):
                Fsvf += self.apply_grid_sample(
                    Fsvf, self.grid + self.as_channel_last(Fsvf)
                )

            Fsvf_neg = -F * steplength
            for _ in torch.arange(self.n_steps):
                Fsvf_neg += self.apply_grid_sample(
                    Fsvf_neg, self.grid + self.as_channel_last(Fsvf_neg)
                )

            F = Fsvf
            Fneg = Fsvf_neg
        else:
            Fneg = None

        self.nonlinear_transform_forward = F
        self.nonlinear_transform_backward = Fneg


    def forward(self, grid):
        nonlin_deform = self.as_channel_last(self.nonlinear_transform_fwd)
        return grid + self.nonlinear_transform_forward


class Resize(torch.nn.Module):
    def __init__(self, size: torch.Tensor | tuple | list, mode = "bilinear"):
        self.size = tuple(size)
        self.mode = mode or "bilinear" # trilinear when `image` is 5-D

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


class RandFlipImage(RandomizableTransform):
    def __init__(self, prob: float = 1):
        super().__init__(prob)

    def forward(self):

class RandFlipSurface(RandomizableTransform):
    def __init__(self, prob: float = 1):
        super().__init__(prob)

    def forward(self):



class GridSample(torch.nn.Module):

    def __init__(self):


    @staticmethod
    def center_from_shape(shape):
        return 0.5 * (shape - 1.0) # -1.0 because of align_corners=True


    def normalize_coordinates(self, v, shape):
        c = self.center_from_shape(shape)
        return (v - c) / c

    def forward(self, image, grid, **kwargs):
        """If the grid is None just apply the spatial cropper"""
        # image : (C, W, H, D)
        # grid  : (V, 3) or (W, H, D, 3)
        if grid is None:
            return image

        idims = len(image.shape)
        gdims = len(grid.shape)
        assert idims in {3, 4}
        assert gdims in {2, 4}

        shape = torch.tensor(image.shape[-3:])

        # image
        # [n,c,]x,y,z -> n,c,z,y,x (!)
        if idims == 3:
            image = image[None, None]
        elif idims == 4:
            image = image[None]
        image = image.transpose(2, 4)  # NOTE xyz -> zyx (!)

        # grid
        norm_grid = self.normalize_coordinates(grid, shape)
        if gdims == 2:
            norm_grid = norm_grid[None, :, None, None]  # 1,W,1,1,3
        elif gdims == 4:
            norm_grid = norm_grid[None]  # 1,W,H,D,3

        # NOTE sample has original xyz ordering! (N,C,W,H,D)
        sample = torch.nn.functional.grid_sample(
            image, norm_grid, align_corners=True, **kwargs
        )
        if gdims == 2:
            sample.squeeze_((2, 3))
        sample.squeeze_(0)

        return sample

class RandResolution(RandomizableTransform):
    """Simulate a"""
    def __init__(self, out_size, in_res, out_res):

        self.input_resolution = in_res
        self.target_resolution = out_res

        self.resize = Resize(out_size)

        if not self.input_resolution.equal(self.target_resolution):
            super().randomize()
            if self._do_transform:

                out_res, thickness = self.random_sampler(self.input_resolution)

                stds = (
                    (0.85 + 0.3 * torch.rand(1.0))
                    * torch.log(torch.tensor([5.0]))
                    / torch.pi
                    * thickness
                    / self.input_resolution
                )
                # no blur if thickness is equal to the resolution of the training
                # data
                stds[thickness <= self.input_resolution] = 0.0

                self.stds = stds
            else:
                self.stds = None
        else:
            self.stds = None


    def forward(self, image):

        if self.stds is None:
            return image



        # Apply resolution and slice thickness blurring if different from the
        # target resolution
        if eq_res := self.input_resolution.equal(self.target_resolution):
            SYN_small = image
        else:

            SYN_blurred = monai.transforms.GaussianSmooth(stds)(image)

            new_size = (self.fov_size * input_resolution / target_resolution).to(torch.int)

            factors = new_size / self.fov_size
            delta = (1.0 - factors) / (2.0 * factors)
            hv = tuple(
                torch.arange(d, d + ns / f, 1 / f)[:ns]
                for d, ns, f in zip(delta, new_size, factors)
            )
            small_grid = torch.stack(torch.meshgrid(*hv, indexing="ij"), dim=-1)

            # the downsampled, synthetic image
            SYN_small = self.apply_grid_sample(SYN_blurred, small_grid)


        SYN_noisy = SYN_small
        # SYN_noisy = self.rand_gauss_noise(SYN_small)

        # is this necessary?
        SYN_noisy = self.eliminate_negative_values(SYN_noisy)


        SYN_resized = self.resize_to_fov(SYN_noisy) if not eq_res else SYN_noisy

        return SYN_resized

        SYN_final = self.normalize_intensity(SYN_resized)

        return SYN_final