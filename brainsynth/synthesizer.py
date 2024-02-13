import warnings

import torch
import monai

from brainsynth.config.utilities import load_config
from brainsynth.constants import constants
from brainsynth.constants.FreeSurferLUT import FreeSurferLUT as lut
from brainsynth.spatial_utils import get_roi_center_size
from brainsynth.transforms import AsDiscreteWithReindex

from brainsynth.supersynth_utils import make_affine_matrix, resolution_sampler, resolution_sampler_1mm_isotropic
HEMI_FLIP = dict(zip(constants.HEMISPHERES, constants.HEMISPHERES[::-1]))


class Synthesizer(torch.nn.Module):
    def __init__(self, config=None, ensure_divisible_by=None, device="cpu"):
        """Dataset synthesizer.


        Parameters
        ----------
        config : dict
        device : str
        """
        super().__init__()

        self.config = config or load_config()
        self.device = device
        self.fov_size_divisor = ensure_divisible_by

        # Map to tensors and device
        match self.config.fov.size:
            case str():
                # In this case, we need to infer FOV size dynamically on call
                raise NotImplementedError(
                    "Currently, you have to explicitly specify FOV size"
                )
                self.fov_size = None
                self.static_fov = False
            case _:
                self.fov_size = torch.tensor(self.config.fov.size, device=device)
                self.fov_size += 2 * torch.tensor(
                    self.config.fov.pad, device=device
                ).expand(3)
                self.static_fov = True

        if self.config.rng_seed is not None:
            torch.manual_seed(self.config.rng_seed)

        self.initialize_state()

        self.linear_transform = None
        self.scale_distance = 1.0
        self.nonlinear_transform_fwd = None
        self.nonlinear_transform_bwd = None

        # Transformations

        self.as_channel_last = monai.transforms.AsChannelLast()

        # Spatial
        self.flip_lr = monai.transforms.Flip(spatial_axis=-3)  # assume C,W,H,D
        # self.flip_lr = monai.transforms.RandFlip(prob=, spatial_axis=1)

        # self.center_spatial_crop = monai.transforms.CenterSpatialCrop()

        # spatial
        # self.rand_affine = monai.transforms.RandAffine(
        #     prob = 1.0, # config.deformation.linear.probability,
        #     rotate_range = torch.deg2rad(
        #         torch.tensor(config.deformation.linear.rotation_range)
        #     ),
        #     shear_range = config.deformation.linear.shear_range,
        #     translate_range = 0, # we set translate seperately as FOV center
        #     scale_range = config.deformation.linear.scale_range,
        # )

        # self.rand_grid = monai.transforms.RandGrid(prob=1, )

        # intensity

        # torch.exp(self.config["intensity"]["gamma_std"] * torch.Tensor([2.0]))

        # gamma: previous sampled from normal distribution around 1.0
        cfg = self.config.intensity.gamma_transform
        self.gamma_transform = monai.transforms.RandAdjustContrast(
            cfg.probability, cfg.gamma_range
        )

        # Synthesized image

        # self.blabla = monai.transforms.compose([])

        self.scale_intensity = monai.transforms.ScaleIntensity()  # scale to 0-1
        self.eliminate_negative_values = monai.transforms.ThresholdIntensity(
            0.0, cval=0.0
        )

        # artifacts

        # self.artifacts = monai.transforms.compose([])
        cfg = self.config.intensity.gaussian_noise
        self.rand_gauss_noise = monai.transforms.RandGaussianNoise(
            cfg.probability,
            cfg.mean,
            cfg.std,
        )
        # self.gibbs = monai.transforms.RandGibbsNoise()
        # self.alias =
        # self.motion =
        # self.biasfield =

        with torch.device(self.device):
            self.set_grid_from_fov(self.fov_size)
            self._complete_setup()

    def _complete_setup(self):
        # new image grid
        # self.center = (self.fov_size - 1) / 2
        # self.grid_centered = monai.transforms.utils.create_grid(
        #     self.fov_size, dtype=torch.float, device=self.device, backend=TransformBackends.TORCH
        # )
        # self.grid = self.grid_centered.clone()
        # self.grid[:3] += self.center[:, None, None, None]

        # seg_labels = torch.tensor(self.config.labeling.synth.incl_csf)
        seg_labels = torch.tensor(
            getattr(constants.labeling_scheme, self.config.labeling_scheme)
        )
        n_seg_labels = len(seg_labels)
        self.as_onehot = AsDiscreteWithReindex(seg_labels)

        # n_neutral_labels = self.config.labeling.synth.incl_csf_n_neutral_labels
        n_neutral_labels = getattr(
            constants.n_neutral_labels, self.config.labeling_scheme
        )
        nlat = int((n_seg_labels - n_neutral_labels) / 2.0)
        self.vflip = torch.cat(
            (
                torch.arange(n_neutral_labels),
                torch.arange(n_neutral_labels + nlat, n_seg_labels),
                torch.arange(n_neutral_labels, n_neutral_labels + nlat),
            )
        )

    def ensure_divisible_size(self, size):
        if self.fov_size_divisor is not None:
            min_size = size / self.fov_size_divisor
            if (residual := torch.ceil(min_size) - min_size).any():
                return (size + self.fov_size_divisor * residual).to(size.dtype)
        return size

    @staticmethod
    def center_from_shape(shape):
        return 0.5 * (shape - 1.0)

    def normalize_coordinates(self, v, shape):
        c = self.center_from_shape(shape)
        return torch.clip((v - c) / c, -1.0, 1.0)

        # when using align_corners=False
        # normalizer = (2.0 / torch.maximum(torch.Tensor([2.0]), self.fov_size[:, None, None, None]))
        # return torch.clip(self.grid * normalizer, -1.0, 1.0)

    # def prepare_grid(self):

    #     self.center = self.center_from_shape(self.fov_size)
    #     self.grid_center = monai.transforms.utils.create_grid(
    #         self.fov_size,
    #         # homogeneous=False,
    #         backend=monai.utils.enums.TransformBackends.TORCH
    #     )

    # normalized coords for align_corners=True: grid / center

    # NOTE Resample's automatic normalization only makes sense for align_corners=False
    # resample = monai.transforms.Resample()

    def set_grid_from_fov(self, fov_size=None):
        if fov_size is None:
            self.grid = None
            self.center = None
            self.grid_center = None
            self.resize_to_fov = None
        else:
            self.grid = torch.stack(
                torch.meshgrid(
                    [torch.arange(s, dtype=torch.float) for s in fov_size],
                    indexing="ij",
                ),
                dim=-1,
            )
            self.center = self.center_from_shape(fov_size)
            self.grid_center = self.grid - self.center
            self.resize_to_fov = monai.transforms.Resize(
                fov_size, mode=self.config.fov.resize_order
            )

    def check_prob(self, probability):
        """Return True according to `probability`."""
        return (probability == 1.0) or (torch.rand(1, device=self.device) < probability)
        # monai.transforms.utils.rand_choice(probability)

    # def set_photo_mode(self, force_state=None):
    #     if force_state is None:
    #         self.photo_mode = self.dos["photo_mode"]
    #     else:
    #         self.photo_mode = force_state

    def set_photo_mode_spacing(self):
        self.spacing = 2.0 + 10 * torch.rand(1) if self.do_task["photo_mode"] else None

    def get_roi_center_size(self, surfaces, bbox, img_shape):
        """


        roi_center
        roi_size
        """

        # Set the center of the deformed grid (i.e., the translation
        # parameter of the affine component)
        if self.config.fov.center == "random_hemi":
            # choose a random hemisphere
            center = tuple(surfaces.keys())[torch.randint(0, len(surfaces), (1,))]
        elif self.config.fov.center == "random":
            raise NotImplementedError

            # IF RANDOM CENTER ...
            # # sample new center
            # if self.config.fov.center == "random":
            #
            #   max_shift = torch.clamp((img_shape - self.fov_size) / 2, 0)
            #   deformed_origin = vol_center + (
            #       2 * (max_shift * torch.rand(3)) - max_shift
            #   )
        else:
            center = self.config.fov.center

        return get_roi_center_size(
            center,
            self.fov_size,  # if self.fov_size is not None else self.config.fov.size,
            self.config.fov.pad,
            bbox,
            img_shape,
        )

    def remove_batch_dim_(self, data):
        for k, v in data.items():
            if isinstance(v, monai.data.MetaTensor):
                # assert v.shape[0] == 1, "Only a batch size of 1 is supported"
                data[k] = v[0] if v.is_batch else v
            else:
                self.remove_batch_dim_(data[k])

    def add_batch_dim_(self, data):
        for k, v in data.items():
            if isinstance(v, monai.data.MetaTensor):
                data[k] = v if v.is_batch else v[None]
            else:
                self.add_batch_dim_(data[k])

    # def add_batch_dim_surface(self, data):
    #     # ADD BACK BATCH DIM !
    #     for hemi, vv in data.items():
    #         for surf, vvv in vv.items():
    #             # if surf != "faces":
    #             data[hemi][surf] = vvv[None]

    def initialize_state(self):
        self.do_task = dict(
            bag = self.check_prob(self.config.exvivo.bag.probability),
            exvivo = self.check_prob(self.config.exvivo.probability),
            flip = self.check_prob(self.config.flip.probability),
            intensity = self.check_prob(self.config.intensity.probability),
            skull_strip = self.check_prob(self.config.intensity.skull_strip.probability),
            linear_deform = self.check_prob(self.config.deformation.linear.probability),
            nonlinear_deform = self.check_prob(
                self.config.deformation.nonlinear.probability
            ),
            photo_mode = self.check_prob(self.config.photo_mode.probability),
        )
        self.set_photo_mode_spacing()

    def forward(self, images, true_surfs, init_surfs, info, disable_synth=False):
        """

        Only batch_size = 1 is supported!

        Parameters
        ----------
        images : _type_
            _description_
        true_surfs : _type_
            _description_
        init_surfs : _type_
            _description_
        info : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # template_surface = template_surface or {}

        with torch.device(self.device):
            # ugly hack for now...

            # remove batch dim from info tensors...
            for k, v in info.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        info[k][kk] = vv.squeeze()
                else:
                    info[k] = v.squeeze()

            # HANDLE THIS INTERNALLY INSTEAD
            self.remove_batch_dim_(images)
            self.remove_batch_dim_(true_surfs)
            self.remove_batch_dim_(init_surfs)

            self.initialize_state()

            if disable_synth:
                self.do_task["intensity"] = False

            original_shape = info["shape"]
            self.has_surfaces = len(true_surfs) > 0

            roi_center, roi_size = self.get_roi_center_size(
                true_surfs, info["bbox"], original_shape
            )

            if not self.static_fov:
                self.set_grid_from_fov(roi_size)

            deformed_origin = roi_center

            # The first thing we do is sample the resolution and deformation,
            # as this will give us a bounding box
            # of the image region we need, so we don't have to read the whole thing from disk (only works for uncompressed niftis!

            self.randomize_linear_transform()
            self.randomize_nonlinear_transform()

            self.construct_deformed_grid(original_shape, deformed_origin)

            # Process images
            images = self.crop_images(images)
            images = self.preprocess_images(images)
            deformed = self.transform_images(images)
            deformed = self.postprocess_images_(deformed)
            deformed |= self.synthesize_image(images["generation"], info["resolution"])
            deformed = self.flip_images(deformed)

            out_shape = next(iter(deformed.values())).shape

            # Process surfaces
            true_surfs = self.transform_surfaces(true_surfs, deformed_origin)
            true_surfs = self.flip_surfaces(true_surfs, out_shape)

            # Process template surfaces
            init_surfs = self.transform_surfaces(init_surfs, deformed_origin)
            init_surfs = self.flip_surfaces(init_surfs, out_shape)

            # HANDLE THIS INTERNALLY INSTEAD

            # only add if they were removed...

            self.add_batch_dim_(deformed)
            self.add_batch_dim_(true_surfs)
            self.add_batch_dim_(init_surfs)

            return deformed, true_surfs, init_surfs

    # def randomize_linear_transform(self):
    #     # By default RandAffine generates a new random affine *each time* it is
    #     # called/applied. We just want a single, random affine, so we feed it
    #     # to Affine
    #     if not self.check_prob(self.config.deformation.linear.probability):
    #         return None, 1.0
    #     else:
    #         self.rand_affine.randomize() # generate new parameters

    #         # distance scaling: we divide distance maps by this, not perfect,
    #         # but better than nothing
    #         scale_distances = torch.prod(self.rand_affine.rand_affine_grid.scale_params) ** (1/3)

    #         return monai.transforms.Affine(
    #             self.rand_affine.rand_affine_grid.rotate_params,
    #             self.rand_affine.rand_affine_grid.shear_params,
    #             self.rand_affine.rand_affine_grid.translate_params,
    #             self.rand_affine.rand_affine_grid.scale_params,
    #         ), scale_distances

    def randomize_linear_transform(self):
        cfg = self.config.deformation.linear

        if not self.do_task["linear_deform"]:
            self.linear_transform = None
            self.scale_distance = 1.0
            return

        rotations = (
            (2 * cfg.max_rotation * torch.rand(3) - cfg.max_rotation) / 180.0 * torch.pi
        )
        shears = 2 * cfg.max_shear * torch.rand(3) - cfg.max_shear
        scalings = 1 + (2 * cfg.max_scale * torch.rand(3) - cfg.max_scale)
        # we divide distance maps by this, not perfect, but better than nothing
        scaling_factor_distances = torch.prod(scalings) ** 0.33333333333

        self.linear_transform = make_affine_matrix(
            rotations, shears, scalings, self.device
        )
        self.scale_distance = scaling_factor_distances

    def apply_grid_sample(self, image, grid, **kwargs):
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

    def randomize_nonlinear_transform(self):
        """
        To be consistent with torch convention that the channel dimension is
        before the spatial dimensions, we create the deformation field such
        that the deformations are [C,W,H,D] although when applying it in
        `compute_deformed_grid` we use it as [W,H,D,C].

        """
        cfg = self.config.deformation.nonlinear

        if not self.do_task["nonlinear_deform"]:
            self.nonlinear_transform_fwd = None
            self.nonlinear_transform_bwd = None
            return

        nonlin_scale = cfg.nonlin_scale_min + torch.rand(1) * (
            cfg.nonlin_scale_max - cfg.nonlin_scale_min
        )
        size_F_small = torch.round(nonlin_scale * self.fov_size).to(torch.int)
        if self.do_task["photo_mode"]:
            size_F_small[1] = torch.round(self.fov_size[1] / self.spacing).to(
                size_F_small.dtype
            )
        nonlin_std = cfg.nonlin_std_max * torch.rand(1)

        Fsmall = nonlin_std * torch.randn([3, *size_F_small])
        F = self.resize_to_fov(Fsmall)

        if self.do_task["photo_mode"]:
            F[1] = 0

        if (
            self.has_surfaces and self.config.surfaces.resolution is not None
        ):  # NOTE: slow
            steplength = 1.0 / (2.0**cfg.n_steps_svf_integration)
            Fsvf = F * steplength
            for _ in torch.arange(cfg.n_steps_svf_integration):
                Fsvf += self.apply_grid_sample(
                    Fsvf, self.grid + self.as_channel_last(Fsvf)
                )

            Fsvf_neg = -F * steplength
            for _ in torch.arange(cfg.n_steps_svf_integration):
                Fsvf_neg += self.apply_grid_sample(
                    Fsvf_neg, self.grid + self.as_channel_last(Fsvf_neg)
                )

            F = Fsvf
            Fneg = Fsvf_neg
        else:
            Fneg = None

        self.nonlinear_transform_fwd = F
        self.nonlinear_transform_bwd = Fneg

    def adjust_roi_center(self, shape: torch.Tensor):
        """Adjust ROI center such that the spatial cropping results in
        the expected size. This might not happen if the slices extend
        beyond the image size (e.g., less than zero, larger than 256)
        in which case we will move the ROI center."""
        # shape = torch.tensor(shape)

        halfsize = 0.5 * self.roi_size
        roi_start = (self.roi_center - halfsize).to(self.roi_center.dtype)
        roi_end = (self.roi_center + halfsize).to(self.roi_center.dtype)

        # we could just set roi as image center and size as image size
        assert torch.all(
            roi_end - roi_start <= shape
        ), f"ROI ({roi_end - roi_start}) is larger than image ({shape})"

        move_roi_center = torch.zeros_like(self.roi_center)

        # check less than 0
        outside = roi_start < 0
        move_roi_center[outside] -= roi_start[outside]

        # check larger than image shape
        mm = (roi_end_d := roi_end - shape) > 0
        move_roi_center[mm] -= roi_end_d[mm]

        self.roi_center += move_roi_center

    def construct_deformed_grid(self, shape, center_coords):
        if not self.do_task["linear_deform"] and not self.do_task["nonlinear_deform"]:
            self.deformed_grid = None
            # make a spatial cropper that includes the to-be-sampled voxels
            self.roi_center = center_coords
            self.roi_size = self.ensure_divisible_size(self.fov_size)
            self.adjust_roi_center(shape)

            self.spatial_crop = monai.transforms.SpatialCrop(
                roi_center=self.roi_center,
                roi_size=self.roi_size,
            )
            return

        deformed_grid = self.grid_center.clone()  # the centered grid

        if self.do_task["nonlinear_deform"]:
            nonlin_deform = self.as_channel_last(self.nonlinear_transform_fwd)
            # avoid in-place modifications
            deformed_grid += nonlin_deform

        if self.do_task["linear_deform"]:
            # deformed_grid = self.affine_grid(deformed_grid)
            deformed_grid = deformed_grid @ self.linear_transform.T

        if center_coords is not None:
            deformed_grid += center_coords

        for i, s in enumerate(shape):
            deformed_grid[..., i].clamp_(0, s - 1)

        margin_min = deformed_grid.amin((0, 1, 2)).floor()
        margin_max = deformed_grid.amax((0, 1, 2)).ceil() + 1

        self.roi_size = self.ensure_divisible_size(margin_max - margin_min)
        self.roi_center = self.center_from_shape(self.roi_size)
        self.adjust_roi_center(shape)

        # make a spatial cropper that includes the to-be-sampled voxels
        self.spatial_crop = monai.transforms.SpatialCrop(
            roi_center=self.roi_center,
            roi_size=self.roi_size,
        )
        # self.spatial_crop = monai.transforms.SpatialCrop(
        #     roi_start=margin_min, roi_end=margin_max
        # )

        self.deformed_grid = deformed_grid - margin_min

        # if nonlin_deform is None and lin_deform is None:
        #     return None
        # else:
        #     # update the grid accordingly
        #     return deformed_grid - margin_min

    def crop_images(self, images):
        return {k: self.spatial_crop(v) for k, v in images.items()}

    def preprocess_images(
        self,
        images: dict,
    ):
        # assert "generation" in images, "generation image is needed for synthesis"

        # if (k := "norm") in images:
        #     # Normalize values
        #     if "generation" in images:
        #         images[k].clamp_(0, None)
        #         images[k] /= torch.median(
        #             images[k][images["generation"] == lut.Left_Cerebral_White_Matter]
        #         )
        #     else:
        #         images[k] = self.scale_intensity(images[k])

        if (k := "bag") in images:
            images[k] /= self.scale_distance

        if (k := "distances") in images:
            images[k] /= self.scale_distance

        # Decide if we're simulating ex vivo (and possibly a bag) or photos
        if self.do_task["photo_mode"] or self.do_task["exvivo"]:
            gen_label = images["generation"]
            gen_label[gen_label > 255] = lut.Unknown  # kill extracerebral
            if self.do_task["photo_mode"]:
                seg_label = images["segmentation"]
                test_elements = torch.IntTensor(
                    [
                        lut.Left_Cerebellum_White_Matter,
                        lut.Left_Cerebellum_Cortex,
                        lut.Brain_Stem,
                    ]
                )
                gen_label[torch.isin(gen_label, test_elements)] = lut.Unknown

                test_elements = torch.IntTensor(
                    [
                        lut.Left_Cerebellum_White_Matter,
                        lut.Left_Cerebellum_Cortex,
                        lut.Ventricle_4th,
                        lut.Brain_Stem,
                        lut.CSF,
                        lut.Right_Cerebellum_White_Matter,
                        lut.Right_Cerebellum_Cortex,
                    ]
                )
                seg_label[torch.isin(seg_label, test_elements)] = lut.Unknown

                # without distance maps, killing 4 is the best we can do
                if "distances" not in images:
                    gen_label[gen_label == lut.Left_Lateral_Ventricle] = lut.Unknown
                else:
                    # pial surfaces are 1 an 3
                    Dpial = torch.minimum(
                        images["distances"][1], images["distances"][3]
                    )
                    th = 1.5 * torch.rand(1)  # band of random width...
                    gen_label[gen_label == lut.Left_Lateral_Ventricle] = 0
                    gen_label[
                        (gen_label == lut.Unknown) & (Dpial < th)
                    ] = lut.Left_Lateral_Ventricle

            elif ("bag" in images) and self.do_task["bag"]:
                gen_shape = torch.as_tensor(gen_label.shape)
                bag_scale = self.config.exvivo.bag.scale_min + torch.rand(1) * (
                    self.config.exvivo.bag.scale_max - self.config.exvivo.bag.scale_min
                )
                size_TH_small = torch.round(bag_scale * gen_shape).int()
                size_TH_small[0] = gen_label.shape[0]
                bag_tness = torch.sort(1.0 + 20 * torch.rand(2)).values
                THsmall = bag_tness[0] + (bag_tness[1] - bag_tness[0]) * torch.rand(
                    *size_TH_small
                )
                TH = monai.transforms.Resize(gen_shape[1:], mode="trilinear")(THsmall)

                gen_label[
                    (images["bag"] > lut.Unknown) & (images["bag"] < TH)
                ] = lut.Left_Lateral_Ventricle

        return images

    def transform_surface_(self, v, translation, A_inv):
        v = v - translation
        v = v @ A_inv.T if A_inv is not None else v
        v += self.center
        if self.nonlinear_transform_bwd is not None:
            v += self.apply_grid_sample(self.nonlinear_transform_bwd, v).squeeze().T
        if (v.amin() < 0) or torch.any(v.amax(0) > self.fov_size):
            warnings.warn(
                (
                    "Cortical surface is partly outside of FOV. BBOX of FOV is "
                    f"{(0,0,0), self.fov_size}. BBOX of surface is "
                    f"{v.amin(0), v.amax(0)}"
                )
            )
        return v

    def transform_surfaces(self, surfaces, deformed_origin):
        if not self.has_surfaces:
            return {}

        inv_lin_deform = (
            torch.linalg.inv(self.linear_transform)
            if self.linear_transform is not None
            else None
        )

        surfaces = {
            h: {
                s: self.transform_surface_(
                    surfaces[h][s],
                    deformed_origin,
                    inv_lin_deform,
                )
                for s in surfs
            }
            if isinstance(surfs, dict)
            else self.transform_surface_(
                surfs,
                deformed_origin,
                inv_lin_deform,
            )
            for h, surfs in surfaces.items()
        }

        return surfaces

    def generate_gaussians(self):
        # extracerebral tissue (from k-means clustering) are added as
        # 500 + label so we need to take this into account, hence the value of
        # n
        cfg = self.config.intensity
        n = 500 + 10
        mu = cfg.mu_offset + cfg.mu_scale * torch.rand(n)
        sigma = cfg.sigma_offset + cfg.sigma_scale * torch.rand(n)

        # set the background to zero every once in a while (or always in photo mode)
        if self.do_task["photo_mode"] or torch.rand(1) < 0.5:
            mu[0] = 0

        return mu, sigma

    def synthesize_contrast(self, label):
        Gr = label.round().long()

        mu, sigma = self.generate_gaussians()

        # skull-strip every now and then by relabeling extracerebral tissues to
        # background
        if self.do_task["skull_strip"]:
            Gr[Gr >= 500] = 0

        # sample synthetic image from gaussians
        SYN = mu[Gr] + sigma[Gr] * torch.randn(Gr.shape)

        if self.config.intensity.partial_volume_effect:
            mask = label != Gr
            SYN[mask] = 0
            Gv = label[mask]
            isv = torch.zeros(Gv.shape)
            pw = (Gv <= 3) * (3 - Gv)
            isv += pw * mu[2] + pw * sigma[2] * torch.randn(Gv.shape)
            pg = (Gv <= 3) * (Gv - 2) + (Gv > 3) * (4 - Gv)
            isv += pg * mu[3] + pg * sigma[3] * torch.randn(Gv.shape)
            pcsf = (Gv >= 3) * (Gv - 3)
            isv += pcsf * mu[4] + pcsf * sigma[4] * torch.randn(Gv.shape)
            SYN[mask] = isv

        SYN[SYN < lut.Unknown] = lut.Unknown

        return SYN

    def transform_images(self, images):
        deformed = {}
        for name, img in images.items():
            match name:
                case "generation":
                    pass
                case "segmentation":
                    # if self.config.deformation.deform_one_hots:
                    #     onehot = self.as_onehot(img)
                    #     onehot = self.apply_grid_sample(onehot, self.deformed_grid)
                    # else:
                    #     def_labels = self.apply_grid_sample(
                    #         img.float(), self.deformed_grid, mode="nearest"
                    #     )
                    #     onehot = self.as_onehot(def_labels)
                    # deformed["label"] = def_labels  # this should be handled...
                    # deformed["segmentation"] = onehot

                    deformed[name] = self.apply_grid_sample(
                        img.float(), self.deformed_grid, mode="nearest"
                    )
                case "distances":
                    # NOTE padding mode in Eugenio's version was distances.amax()
                    deformed[name] = self.apply_grid_sample(
                        img, self.deformed_grid, padding_mode="border"
                    )
                case _:
                    # could also include pathologies...
                    deformed[name] = self.apply_grid_sample(img, self.deformed_grid)

        return deformed

    def postprocess_images_(self, images):
        """In-place"""
        if "segmentation" in images:
            images["segmentation"] = self.as_onehot(images["segmentation"])

        for name in images:
            match name:
                case "norm":
                    # Remove background from real images
                    # NOTE Perhaps we shouldn't do this???
                    images["norm"] *= 1.0 - images["segmentation"][lut.Unknown]

        return images

    def synthesize_image(
        self,
        gen_label: torch.Tensor,
        input_resolution: torch.Tensor,
    ):
        if not self.do_task["intensity"]:
            return {}

        # Generate (log-transformed) bias field

        biasfield = self.synthesize_bias_field()
        # biasfield = monai.transforms.RandBiasField(degree=5, coeff_range=(0,0.1), prob=0.5)

        # Synthesize the new contrast and deform it
        synth = self.apply_grid_sample(
            self.synthesize_contrast(gen_label), self.deformed_grid
        )

        # Manipulate synthesizer image

        # deformed["synth_clean"] = deformed["synth"].clone().detach()
        synth = self.gamma_transform(synth)
        synth = self.add_bias_field(synth, biasfield)
        synth = self.simulate_resolution(synth, input_resolution)

        return dict(biasfield=biasfield, synth=synth)

    def flip_images(self, deformed):
        if not self.do_task["flip"]:
            return deformed

        for k, v in deformed.items():
            deformed[k] = self.flip_lr(v)
            if k == "segmentation":
                deformed[k] = deformed[k][self.vflip]
            elif k == "distances":
                deformed[k] = deformed[k][[2, 3, 0, 1]]  # flip left/right

        return deformed

    def flip_surfaces(self, surfaces, shape):
        if not self.has_surfaces or not self.do_task["flip"]:
            return surfaces

        def _flip_surface_coords(s, width):
            s[:, 0] *= -1.0
            s[:, 0] += width - 1.0

        width = shape[1]  # assumes C,W,H,D
        for hemi, surfs in surfaces.items():
            if isinstance(surfs, dict):
                for s in surfs:
                    _flip_surface_coords(surfaces[hemi][s], width)
            else:
                _flip_surface_coords(surfs, width)

        # NOTE
        # swap label (e.g., lh -> rh) when flipping image. Not sure if
        # this is the best thing to do but at least ensures some kind
        # of consistency: when flipping then faces need to be reordered
        # which is also the case for rh compared to lh.
        return {HEMI_FLIP[h]: v for h, v in surfaces.items()}

    def synthesize_bias_field(self):
        """Synthesize a log-transformed bias field."""
        bf = self.config.biasfield

        bf_scale = bf.scale_min + torch.rand(1) * (bf.scale_max - bf.scale_min)
        size_BF_small = torch.round(bf_scale * self.fov_size).to(torch.int)

        if self.do_task["photo_mode"]:
            size_BF_small[1] = torch.round(self.fov_size[1] / self.spacing).to(
                torch.int
            )

        BFsmall = bf.std_min + (bf.std_max - bf.std_min) * torch.rand(1) * torch.randn(
            *size_BF_small
        )

        return self.resize_to_fov(BFsmall[None])  # add channel dim

    def add_bias_field(self, image, bias_field):
        """Add a log-transformed bias field to an image."""
        # factor = 300.0
        # # Gamma transform
        # gamma = torch.exp(self.config["intensity"]["gamma_std"] * torch.randn(1))
        # SYN_gamma = factor * (image / factor) ** gamma
        return image * bias_field.exp()

    def simulate_resolution(self, image, input_resolution):
        """Simulate a"""

        target_resolution, thickness = self.random_sampler(input_resolution)

        # Apply resolution and slice thickness blurring if different from the
        # target resolution
        if input_resolution.equal(target_resolution):
            SYN_small = image
        else:
            stds = (
                (0.85 + 0.3 * torch.rand(1.0))
                * torch.log(torch.tensor([5.0]))
                / torch.pi
                * thickness
                / input_resolution
            )
            # no blur if thickness is equal to the resolution of the training
            # data
            stds[thickness <= input_resolution] = 0.0

            SYN_blurred = monai.transforms.GaussianSmooth(stds)(image)

            new_size = (self.fov_size * input_resolution / target_resolution).to(torch.int)

            factors = new_size / self.fov_size
            delta = (1.0 - factors) / (2.0 * factors)
            hv = tuple(
                torch.arange(d, d + ns / f, 1 / f)[:ns]
                for d, ns, f in zip(delta, new_size, factors)
            )
            small_grid = torch.stack(torch.meshgrid(*hv, indexing="ij"), dim=-1)

            # the downsampled synthetic image
            SYN_small = self.apply_grid_sample(SYN_blurred, small_grid)

        # noise_std = intensity["noise_std_min"] + (
        #     intensity["noise_std_max"] - intensity["noise_std_min"]
        # ) * torch.rand(1)
        # SYN_noisy = SYN_small + noise_std * torch.randn(SYN_small.shape)

        SYN_noisy = self.rand_gauss_noise(SYN_small)
        SYN_noisy = self.eliminate_negative_values(SYN_noisy)
        SYN_resized = self.resize_to_fov(SYN_noisy)
        SYN_final = self.scale_intensity(SYN_resized)

        return SYN_final

    def random_sampler(self, in_res):
        if self.do_task["photo_mode"]:
            out_res = torch.tensor([in_res[0], self.spacing, in_res[2]])
            thickness = torch.tensor([in_res[0], 0.0001, in_res[2]])
        else:
            out_res = in_res.clone()
            thickness = in_res.clone()
            # out_res, thickness = resolution_sampler_1mm_isotropic(self.device)
        return out_res, thickness
