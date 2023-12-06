import warnings

import torch
import monai

from brainsynth.config.utilities import load_config
from brainsynth.supersynth_utils import make_affine_matrix, resolution_sampler

from brainsynth.constants import constants
from brainsynth.constants.FreeSurferLUT import FreeSurferLUT as lut
from brainsynth.spatial_utils import get_roi_center_size
from brainsynth.transforms import Reindex

HEMI_FLIP = dict(zip(constants.HEMISPHERES, constants.HEMISPHERES[::-1]))

"""
# example

from pathlib import Path
from brainsynth.config.utilities import load_config
from brainsynth import root_dir

base_dir = Path("/mrhome/jesperdn/nobackup/supersynth_data_gen/generator/deformed/new")
subjects = ["sub-01"]
subject_dir = base_dir / subjects[0]
device = "cpu"

config = load_config(root_dir / "config" / "synthesizer.yaml")

dataset = CroppedDataset(
    base_dir,
    subjects,
    images = ("generation", "t1", "segmentation"),
    surface_resolution = 6,
    surface_hemi = "lh",
    fov_center = "lh",
    fov_size = "lh", # (120,200,120),
    fov_pad = 5,
    images_as_one_hot = dict(segmentation=config["labeling"]["synth"]["incl_csf"]),
)

# no cropping
dataset = CroppedDataset(
    base_dir,
    subjects,
    default_images = ("generation", "norm", "segmentation"),
    # surface_resolution = 6,
    surface_hemi = "random",
)
images, surfaces, info = next(iter(dataset))
print(surfaces)

self = Synthesizer(device=device)
data = self(images, surfaces, info)

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

plt.imshow(data["synth"][0,0,:,100])

nib.Nifti1Image(data["t1"].squeeze().numpy(), np.identity(4)).to_filename("/mrhome/jesperdn/nobackup/supersynth_data_gen/generator/deformed/new/sub-01/test_t1.nii")
nib.Nifti1Image(data["synth"].squeeze().numpy(), np.identity(4)).to_filename("/mrhome/jesperdn/nobackup/supersynth_data_gen/generator/deformed/new/sub-01/test_synth.nii")
nib.Nifti1Image(data["label"].squeeze().numpy(), np.identity(4)).to_filename("/mrhome/jesperdn/nobackup/supersynth_data_gen/generator/deformed/new/sub-01/test_label.nii")

v,lhf = nib.freesurfer.read_geometry("/mrhome/jesperdn/nobackup/supersynth_data_gen/generator/deformed/new/sub-01/lh.white")
v,rhf = nib.freesurfer.read_geometry("/mrhome/jesperdn/nobackup/supersynth_data_gen/generator/deformed/new/sub-01/rh.white")



nib.freesurfer.write_geometry("/mrhome/jesperdn/nobackup/supersynth_data_gen/generator/deformed/new/sub-01/test_rh.white",
    data["surfaces"]["rh"]["white"].squeeze().numpy(), rhf)
nib.freesurfer.write_geometry("/mrhome/jesperdn/nobackup/supersynth_data_gen/generator/deformed/new/sub-01/test_rh.pial",
    data["surfaces"]["rh"]["pial"].squeeze().numpy(), rhf)


nib.freesurfer.write_geometry("/mrhome/jesperdn/nobackup/supersynth_data_gen/generator/deformed/new/sub-01/test_rh.white",
    data["surfaces"]["lh"]["white"].squeeze().numpy(), lhf)
nib.freesurfer.write_geometry("/mrhome/jesperdn/nobackup/supersynth_data_gen/generator/deformed/new/sub-01/test_rh.pial",
    data["surfaces"]["lh"]["pial"].squeeze().numpy(), lhf)


"""


class Synthesizer(torch.nn.Module):
    def __init__(self, config=None, device="cpu"):
        """Dataset synthesizer.

        Parameters
        ----------
        config : dict
        device : str
        """
        super().__init__()

        self.config = config or load_config()
        self.device = device

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
        self.as_onehot = monai.transforms.Compose(
            [
                monai.transforms.EnsureType(dtype=torch.int),
                Reindex(seg_labels),
                monai.transforms.AsDiscrete(to_onehot=n_seg_labels),
            ]
        )
        # n_neutral_labels = self.config.labeling.synth.incl_csf_n_neutral_labels
        n_neutral_labels = getattr(constants.n_neutral_labels, self.config.labeling_scheme)
        nlat = int((n_seg_labels - n_neutral_labels) / 2.0)
        self.vflip = torch.cat(
            (
                torch.arange(n_neutral_labels),
                torch.arange(n_neutral_labels + nlat, n_seg_labels),
                torch.arange(n_neutral_labels, n_neutral_labels + nlat),
            )
        )

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

    def set_photo_mode(self, force_state=None):
        if force_state is None:
            self.photo_mode = self.check_prob(self.config.photo_mode.probability)
        else:
            self.photo_mode = force_state
        self.spacing = 2.0 + 10 * torch.rand(1) if self.photo_mode else None

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
        # Remove batch dimension! We add it again later...
        for k, v in data.items():
            if v.ndim == 5:
                data[k] = v[0]

    def add_batch_dim_(self, data):
        # ADD BACK BATCH DIM !
        for k, v in data.items():
            if k == "surfaces":
                for hemi, vv in v.items():
                    for surf, vvv in vv.items():
                        # if surf != "faces":
                        data[k][hemi][surf] = vvv[None]
            else:
                if v.ndim == 4:
                    data[k] = v[None]

    def forward(self, images, surfaces, info):
        with torch.device(self.device):
            input_resolution = info["resolution"]
            img_shape = info["shape"]

            # HANDLE THIS INTERNALLY INSTEAD
            self.remove_batch_dim_(images)

            roi_center, roi_size = self.get_roi_center_size(
                surfaces, info["bbox"], img_shape
            )

            if not self.static_fov:
                self.set_grid_from_fov(roi_size)

            deformed_origin = roi_center

            # The first thing we do is sample the resolution and deformation,
            # as this will give us a bounding box
            # of the image region we need, so we don't have to read the whole thing from disk (only works for uncompressed niftis!

            do_synthesis = self.check_prob(self.config.intensity.probability)

            # Get the deformation parameters
            if do_synthesis:
                self.set_photo_mode()
            else:
                self.set_photo_mode(force_state=False)

            self.randomize_linear_transform()
            self.randomize_nonlinear_transform()
            self.set_deformed_grid(img_shape, deformed_origin)

            images = self.preprocess_images(images)
            surfaces = self.preprocess_surfaces(surfaces, deformed_origin)

            # Synthesize image
            if do_synthesis:
                data = self.process_sample(input_resolution, images, surfaces)
            else:
                data = images
                # data["synth"] = ???
                data["surfaces"] = surfaces

            # HANDLE THIS INTERNALLY INSTEAD
            self.add_batch_dim_(data)

            return data

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

        if not self.check_prob(cfg.probability):
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

        self.linear_transform = make_affine_matrix(rotations, shears, scalings, self.device)
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

        if not self.check_prob(cfg.probability):
            self.nonlinear_transform_fwd = None
            self.nonlinear_transform_bwd = None
            return

        nonlin_scale = cfg.nonlin_scale_min + torch.rand(1) * (
            cfg.nonlin_scale_max - cfg.nonlin_scale_min
        )
        size_F_small = torch.round(nonlin_scale * self.fov_size).to(torch.int)
        if self.photo_mode:
            size_F_small[1] = torch.round(self.fov_size[1] / self.spacing).to(
                size_F_small.dtype
            )
        nonlin_std = cfg.nonlin_std_max * torch.rand(1)

        Fsmall = nonlin_std * torch.randn([3, *size_F_small])
        F = self.resize_to_fov(Fsmall)

        if self.photo_mode:
            F[1] = 0

        if self.config.surfaces.resolution is not None:  # NOTE: slow
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

    def set_deformed_grid(self, shape, center_coords):
        do_linear_transform = self.linear_transform is not None
        do_nonlinear_transform = self.nonlinear_transform_fwd is not None

        if not do_linear_transform or not do_nonlinear_transform:
            self.deformed_grid = None
            # make a spatial cropper that includes the to-be-sampled voxels
            self.spatial_crop = monai.transforms.SpatialCrop(
                center_coords, self.fov_size
            )

        deformed_grid = self.grid_center.clone()  # the centered grid

        if do_nonlinear_transform:
            nonlin_deform = self.as_channel_last(self.nonlinear_transform_fwd)
            # avoid in-place modifications
            deformed_grid += nonlin_deform

        if do_linear_transform:
            # deformed_grid = self.affine_grid(deformed_grid)
            deformed_grid = deformed_grid @ self.linear_transform.T

        if center_coords is not None:
            deformed_grid += center_coords

        for i, s in enumerate(shape):
            deformed_grid[..., i].clamp_(0, s - 1)

        margin_min = deformed_grid.amin((0, 1, 2)).floor()
        margin_max = deformed_grid.amax((0, 1, 2)).ceil() + 1

        # make a spatial cropper that includes the to-be-sampled voxels
        self.spatial_crop = monai.transforms.SpatialCrop(
            roi_start=margin_min, roi_end=margin_max
        )

        self.deformed_grid = deformed_grid - margin_min

        # if nonlin_deform is None and lin_deform is None:
        #     return None
        # else:
        #     # update the grid accordingly
        #     return deformed_grid - margin_min

    def apply_spatial_crop_images(self, images):
        return {k: self.spatial_crop(v) for k, v in images.items()}

    def preprocess_images(
        self,
        images: dict,
    ):
        assert "generation" in images, "generation image is needed for synthesis"

        images = self.apply_spatial_crop_images(images)

        gen_label = images["generation"]

        if (k := "segmentation") in images:
            seg_label = images["segmentation"]
        if (k := "bag") in images:
            images[k] /= self.scale_distance
        if (k := "distances") in images:
            images[k] /= self.scale_distance
        if (k := "t1") in images:
            images[k].clamp_(0, None)
            images[k] /= torch.median(
                images[k][images["generation"] == lut.Left_Cerebral_White_Matter]
            )

        # Decide if we're simulating ex vivo (and possibly a bag) or photos
        if self.photo_mode or self.check_prob(self.config.exvivo.probability):
            gen_label[gen_label > 255] = lut.Unknown  # kill extracerebral
            if self.photo_mode:
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

            elif ("bag" in images) and self.check_prob(
                self.config.exvivo.bag.probability
            ):
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

    def preprocess_surfaces(self, surfaces, deformed_origin):
        if len(surfaces) > 0:
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
                    for s in surfaces[h]
                }
                for h in surfaces
            }
            # for s in {"white", "pial"}: # if faces included
        return surfaces

    def generate_gaussians(self):
        # extracerebral tissue (from k-means clustering) are added as
        # 500 + label so we need to take this into account, hence the value of
        # n
        cfg = self.config.intensity
        n = 500 + 10
        mus = cfg.mu_offset + cfg.mu_scale * torch.rand(n)
        sigmas = cfg.sigma_offset + cfg.sigma_scale * torch.rand(n)

        # set the background to zero every once in a while (or always in photo mode)
        if self.photo_mode or torch.rand(1) < 0.5:
            mus[0] = 0

        return mus, sigmas

    def synthesize_contrast(self, label):
        Gr = label.round().long()

        mus, sigmas = self.generate_gaussians()

        # sample synthetic image from gaussians
        SYN = mus[Gr] + sigmas[Gr] * torch.randn(Gr.shape)

        if self.config.intensity.partial_volume_effect:
            mask = label != Gr
            SYN[mask] = 0
            Gv = label[mask]
            isv = torch.zeros(Gv.shape)
            pw = (Gv <= 3) * (3 - Gv)
            isv += pw * mus[2] + pw * sigmas[2] * torch.randn(Gv.shape)
            pg = (Gv <= 3) * (Gv - 2) + (Gv > 3) * (4 - Gv)
            isv += pg * mus[3] + pg * sigmas[3] * torch.randn(Gv.shape)
            pcsf = (Gv >= 3) * (Gv - 3)
            isv += pcsf * mus[4] + pcsf * sigmas[4] * torch.randn(Gv.shape)
            SYN[mask] = isv

        SYN[SYN < lut.Unknown] = lut.Unknown

        return SYN

    def process_sample(
        self,
        input_resolution: torch.Tensor,
        images: dict,
        surfaces: None | dict = None,
        synthesize_contrast: bool = True,
    ):
        images["synth"] = self.synthesize_contrast(images["generation"])

        target_resolution, thickness = self.random_sampler(input_resolution)

        deformed_grid = self.deformed_grid

        # Apply deformation to images
        deformed = {}
        for name, img in images.items():
            match name:
                case "generation":
                    pass
                case "segmentation":
                    if self.config.deformation.deform_one_hots:
                        onehot = self.as_onehot(img)
                        onehot = self.apply_grid_sample(onehot, deformed_grid)
                    else:
                        def_labels = self.apply_grid_sample(
                            img.float(), deformed_grid, mode="nearest"
                        )
                        onehot = self.as_onehot(def_labels)
                    deformed["label"] = def_labels  # this should be handled...
                    deformed["segmentation"] = onehot
                case "distances":
                    # NOTE padding mode in Eugenio's version was distances.amax()
                    deformed[name] = self.apply_grid_sample(
                        img, deformed_grid, padding_mode="border"
                    )
                case _:
                    # could also include pathologies...
                    deformed[name] = self.apply_grid_sample(img, deformed_grid)

        # log transformed bias field
        deformed["biasfield"] = self.synthesize_bias_field()
        # biasfield = monai.transforms.RandBiasField(degree=5, coeff_range=(0,0.1), prob=0.5)

        #
        # deformed["synth_clean"] = deformed["synth"].clone().detach()

        if synthesize_contrast:
            synth = self.gamma_transform(deformed["synth"])
            synth = self.add_bias_field(synth, deformed["biasfield"])
            synth = self.simulate_resolution(
                synth, input_resolution, target_resolution, thickness
            )
            deformed["synth"] = synth

        # Remove background from real images
        if "t1" in deformed:
            deformed["t1"] *= 1.0 - deformed["segmentation"][lut.Unknown]

        deformed, surfaces = self.flip_images_and_surfaces(deformed, surfaces)

        if len(surfaces) > 0:
            deformed["surfaces"] = surfaces

        return deformed

    def flip_images_and_surfaces(self, deformed, surfaces):
        if self.check_prob(self.config.flip.probability):  # Flip 50
            # images
            for k, v in deformed.items():
                deformed[k] = self.flip_lr(v)
                if k == "segmentation":
                    deformed[k] = deformed[k][self.vflip]
                elif k == "distances":
                    deformed[k] = deformed[k][[2, 3, 0, 1]]  # flip left/right

            # surfaces
            if len(surfaces) > 0:
                width = deformed["synth"].shape[1]  # assumes C,W,H,D
                # surfs = {"white", "pial"}
                for hemi, surfs in surfaces.items():  # hemisphere dicts
                    # for s in surfs:
                    # flip coordinates
                    for s in surfs:
                        surfaces[hemi][s][:, 0] *= -1.0
                        surfaces[hemi][s][:, 0] += width - 1.0

                    # flip vertex ordering to preserve normal direction
                    # hv["faces"] = hv["faces"][:, (0, 2, 1)]

                # NOTE
                # swap label (e.g., lh -> rh) when flipping image. Not sure if
                # this is the best thing to do but at least ensures some kind
                # of consistency: when flipping then faces need to be reordered
                # which if also the case when for rh compared to lh.
                surfaces = {HEMI_FLIP[h]: v for h, v in surfaces.items()}

        return deformed, surfaces

    def synthesize_bias_field(self):
        """Synthesize a log-transformed bias field."""
        bf = self.config.biasfield

        bf_scale = bf.scale_min + torch.rand(1) * (bf.scale_max - bf.scale_min)
        size_BF_small = torch.round(bf_scale * self.fov_size).to(torch.int)

        if self.photo_mode:
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

    def simulate_resolution(
        self, image, input_resolution, target_resolution, thickness
    ):
        """Simulate a"""

        # gaussian smooth

        # downsample

        # add gauss noise
        # threshold at > 0
        # upsample
        # normalize by dividing by max intensity

        # Apply resolution and slice thickness blurring if different from the
        # target resolution
        stds = (
            (0.85 + 0.3 * torch.rand(1))
            * torch.log(torch.Tensor([5]))
            / torch.pi
            * thickness
            / input_resolution
        )
        # no blur if thickness is equal to the resolution of the training data
        stds[thickness <= input_resolution] = 0.0

        # SYN_blur = gaussian_blur_3d(image, stds, self.device)

        # self.gaussian_smooth = monai.transforms.GaussianSmooth(stds)
        SYN_blur = monai.transforms.GaussianSmooth(stds)(image)

        # DOWNSAMPLE START >>>

        new_size = (self.fov_size * input_resolution / target_resolution).to(torch.int)

        factors = new_size / self.fov_size
        delta = (1.0 - factors) / (2.0 * factors)
        hv = tuple(
            torch.arange(d, d + ns / f, 1 / f)[:ns]
            for d, ns, f in zip(delta, new_size, factors)
        )
        small_grid = torch.stack(torch.meshgrid(*hv, indexing="ij"), dim=-1)

        # the downsampled synthetic image
        SYN_small = self.apply_grid_sample(SYN_blur, small_grid)

        # DOWNSAMPLE END <<<

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
        if self.photo_mode:
            out_res = torch.Tensor([in_res[0], self.spacing, in_res[2]])
            thickness = torch.Tensor([in_res[0], 0.0001, in_res[2]])
        else:
            out_res, thickness = resolution_sampler(self.device)
        return out_res, thickness
