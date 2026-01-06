from functools import partial
import operator

import torch

from brainsynth.config.synthesizer import SynthesizerConfig
from brainsynth.config.utilities import (
    OutputPipelines,
    Pipelines,
    StatePipelines,
    SynthBuilder,
)
from brainsynth.constants import IMAGE
from brainsynth.transforms import *


class ExvivoSynth(SynthBuilder):
    def __init__(self, config: SynthesizerConfig) -> None:
        """This pipeline includes

        - linear deformation
        - nonlinear deformation
        - synthesis of an image as the input image for training

        """
        super().__init__(config)

    def initialize_spatial_transforms(self):
        """No spatial transformation."""
        # self.linear_transform = IdentityTransform()
        # self.inv_linear_transform = IdentityTransform()
        self.nonlinear_transform = IdentityTransform()
        self.inv_nonlinear_transform = IdentityTransform()

        # self.nonlinear_transform = RandNonlinearTransform(
        #     self.config.out_size,
        #     scale_min=0.03,  # 0.10,
        #     scale_max=0.06,  # 0.15,
        #     std_max=4.0,
        #     exponentiate_field=True,
        #     grid=self.config.grid,
        #     photo_mode=SelectState("photo_mode"),
        #     photo_spacing=SelectState("photo_spacing"),
        #     prob=1.0,
        #     device=self.device,
        # )
        # self.inv_nonlinear_transform = partial(
        #     self.nonlinear_transform, direction="backward"
        # )

        self.linear_transform = RandLinearTransform(
            max_rotation=15.0,
            max_scale=0.1,
            max_shear=0.1,
            prob=0.75,
            device=self.device,
        )
        self.inv_linear_transform = partial(self.linear_transform, inverse=True)

    def build_state(self):
        return StatePipelines(
            # Determine whether we are using photo mode or not
            photo_mode=EvaluateProbability(self.config.photo_mode_prob),
            photo_spacing=Uniform(*self.config.photo_spacing_range, device=self.device),
            available_images=Pipeline(
                SelectImage(), ExtractDictKeys(), unpack_inputs=False
            ),
            in_size=Pipeline(
                PipelineModule(SelectImage, SelectState("available_images", 0)),
                SpatialSize(),
            ),
            surface_bbox=Pipeline(
                SelectSurface("template"), SurfaceBoundingBox(), unpack_inputs=False
            ),
            # mask for cropping to a single hemisphere
            hemi_mask=Pipeline(
                SelectImage("lp_dist_map", "rp_dist_map", mode="all valid"),
                PipelineModule(Concatenate, dim=0),
                PipelineModule(ApplyFunction, partial(torch.amin, dim=0, keepdim=True)),
                PipelineModule(
                    MaskFromFloatImage, 1.5, 3.0, operator.le, device=self.device
                ),
                skip_on_InputSelectorError=True,
            ),
            out_center=Pipeline(
                ServeValue(self.config.out_center_str),
                PipelineModule(
                    CenterFromString,
                    SelectState("in_size"),
                    SelectState("surface_bbox"),
                    align_corners=self.config.align_corners,
                ),
            ),
            resampling_grid=Pipeline(
                ServeValue(self.config.centered_grid),
                self.nonlinear_transform,
                self.linear_transform,
                PipelineModule(
                    TranslationTransform, SelectState("out_center"), device=self.device
                ),
            ),
        )

    def build_spatial_transforms(self):
        self.image_deformation = SubPipeline(
            PipelineModule(
                GridSample,
                SelectState("resampling_grid"),
                SelectState("in_size"),
            ),
        )

        self.surface_deformation = SubPipeline(
            PipelineModule(
                TranslationTransform,
                SelectState("out_center"),
                invert=True,
                device=self.device,
            ),
            self.inv_linear_transform,
            TranslationTransform(self.config.center, device=self.device),
            self.inv_nonlinear_transform,
            CheckCoordsInside(self.config.out_size, device=self.device),
        )
        # (Affine) noise on vertex positions
        self.surface_noise = SubPipeline(
            RandLinearTransform(
                max_rotation=5.0,
                max_scale=0.05,
                max_shear=0.05,
                relative_to_input=True,
                prob=0.5,
                device=self.device,
            ),
            RandTranslationTransform(
                x_range=[-3, 3],
                y_range=[-3, 3],
                z_range=[-3, 3],
                prob=0.5,
                device=self.device,
            ),
        )

    def build_intensity_transforms(self, *args, **kwargs):
        self.mask_hemi = SubPipeline(
            PipelineModule(RandApplyMask, SelectState("hemi_mask"), prob=0.5),
        )
        self.intensity_normalization = IntensityNormalization()

    def build_resolution_transforms(
        self,
        resolution_sampler: str = "ResolutionSamplerDefault",
        resolution_sampler_kw: dict | None = None,
    ):
        resolution_sampler_kw = resolution_sampler_kw or {}

        self.resolution_augmentation = PipelineModule(
            RandResolution,
            self.config.out_size,
            self.config.in_res,
            resolution_sampler,
            resolution_sampler_kw,
            photo_mode=SelectState("photo_mode"),
            photo_spacing=SelectState("photo_spacing"),
            photo_thickness=self.config.photo_thickness,
            prob=0.75,
            device=self.device,
        )
        # self.resolution_augmentation = IdentityTransform()

    def build_image(self):
        return Pipeline(
            # Synthesize image
            SelectImage(self.config.generation_image),
            PipelineModule(
                SynthesizeIntensityImage,
                mu_low=25.0,
                mu_high=200.0,
                sigma_global_scale=4.0,
                sigma_local_scale=2.5,
                # sigma_global_scale=6.0,  # 0.5 mm
                # sigma_local_scale=5.0,  # 0.5 mm
                pv_sigma_range=[0.5, 1.2],
                pv_tail_length=3.0,
                pv_from_distances=False,
                min_cortical_contrast=10.0,
                photo_mode=SelectState("photo_mode"),
                device=self.device,
            ),
            RandGammaTransform(prob=0.33),
            self.mask_hemi,
            self.image_deformation,  # Transform to output FOV
            PipelineModule(
                RandBiasfield,
                self.config.out_size,
                scale_min=0.01,
                scale_max=0.04,
                std=0.3,  # 3T
                # std=0.6,  # 7T
                photo_mode=SelectState("photo_mode"),
                photo_spacing=SelectState("photo_spacing"),
                prob=0.75,
                device=self.device,
            ),
            self.resolution_augmentation,
            self.intensity_normalization,
        )

    def build_affine(self):
        pass

    def build_surface(self):
        return Pipelines(
            white=Pipeline(
                SelectSurface("white"),
                self.surface_deformation,
                skip_on_InputSelectorError=True,
            ),
            pial=Pipeline(
                SelectSurface("pial"),
                self.surface_deformation,
                skip_on_InputSelectorError=True,
            ),
            template=Pipeline(
                SelectSurface("template"),
                self.surface_deformation,
                self.surface_noise,
                skip_on_InputSelectorError=True,
            ),
        )

    def build_images(self):
        return Pipelines(
            t1w=Pipeline(
                SelectImage("t1w"),
                self.mask_hemi,
                self.image_deformation,
                self.intensity_normalization,
                skip_on_InputSelectorError=True,
            ),
            # t1w_mask=Pipeline(
            #     SelectImage("t1w_mask"),
            #     self.mask_hemi,
            #     self.image_deformation,
            #     skip_on_InputSelectorError=True,
            # ),
            brain_dist_map=Pipeline(
                SelectImage("lp_dist_map", "rp_dist_map", mode="first"),
                self.image_deformation,
                skip_on_InputSelectorError=True,
            ),
        )

    def build_output(self):
        return OutputPipelines(
            image=self.build_image(),
            affine=self.build_affine(),
            images=self.build_images(),
            surfaces=self.build_surface(),
        )


class ExvivoSelect(ExvivoSynth):
    """The primary output image is selected from one of the selectable images
    provided.
    """

    def build_state(self):
        state = super().build_state()

        state["selectable_images"] = Pipeline(
            SelectState("available_images"),
            Intersection(self.config.selectable_images),
            unpack_inputs=False,
        )
        return state

    def build_image(self):
        return Pipeline(
            # Select one of the available (valid) images with equal probability
            PipelineModule(
                SelectImage,
                PipelineModule(RandomChoice, SelectState("selectable_images")),
            ),
            self.mask_hemi,
            self.image_deformation,
            self.resolution_augmentation,
            self.intensity_normalization,
        )


class ExvivoSelectIso(ExvivoSelect):
    def build_resolution_transforms(self):
        self.resolution_augmentation = IdentityTransform()


class ExvivoCropSynth(SynthBuilder):
    """Process an image called `image` and"""

    def build_state(self):
        return StatePipelines(
            photo_mode=EvaluateProbability(self.config.photo_mode_prob),
            photo_spacing=Uniform(*self.config.photo_spacing_range, device=self.device),
            available_images=Pipeline(
                SelectImage(), ExtractDictKeys(), unpack_inputs=False
            ),
            in_size=Pipeline(
                PipelineModule(
                    SelectImage,
                    # doesn't matter which is selected - all should be the same size!
                    SelectState("available_images", 0),
                ),
                SpatialSize(),
            ),
            extracerebral_mask=Pipeline(
                SelectImage(self.config.generation_image),
                PipelineModule(
                    MaskFromLabelImage,
                    labels=[0] + list(IMAGE.generation_labels.kmeans),
                    device=self.device,
                ),
            ),
            hemi_mask=Pipeline(
                SelectImage("lp_dist_map", "rp_dist_map", mode="all valid"),
                PipelineModule(Concatenate, dim=0),
                PipelineModule(ApplyFunction, partial(torch.amin, dim=0, keepdim=True)),
                PipelineModule(
                    MaskFromFloatImage, 1.5, 3.0, operator.le, device=self.device
                ),
            ),
            out_center=Pipeline(
                SelectState("hemi_mask"),
                PipelineModule(
                    CenterFromMask,
                    align_corners=self.config.align_corners,
                ),
            ),
            crop_params=Pipeline(
                SelectState("in_size"),
                PipelineModule(
                    SpatialCropParameters,
                    self.config.out_size,
                    SelectState("out_center"),
                ),
            ),
        )

    def build_intensity_transforms(self):
        self.extracerebral_augmentation = SubPipeline(
            PipelineModule(
                SwitchTransform,
                IdentityTransform(),
                # salt and pepper noise (e.g., like MP2RAGE)
                PipelineModule(
                    RandSaltAndPepperNoise,
                    SelectState("extracerebral_mask"),
                    scale=255.0,
                    device=self.device,
                ),
                # hemi mask or skull strip
                PipelineModule(ApplyMask, SelectState("hemi_mask")),
                prob=(0.4, 0.1, 0.5),
            ),
        )
        self.intensity_normalization = IntensityNormalization()

    def build_resolution_transforms(
        self,
        resolution_sampler: str = "ResolutionSamplerDefault",
        resolution_sampler_kw: dict | None = None,
    ):
        resolution_sampler_kw = resolution_sampler_kw or {}

        self.resolution_augmentation = PipelineModule(
            RandResolution,
            self.config.out_size,
            self.config.in_res,
            resolution_sampler,
            resolution_sampler_kw,
            photo_mode=SelectState("photo_mode"),
            photo_spacing=SelectState("photo_spacing"),
            photo_thickness=self.config.photo_thickness,
            prob=0.75,
            device=self.device,
        )
        # self.resolution_augmentation = IdentityTransform()

    def build_spatial_transforms(self):
        self.image_crop = SubPipeline(
            PipelineModule(
                SpatialCrop,
                SelectState("in_size"),
                SelectState("crop_params", "slices"),
            ),
            PipelineModule(
                PadTransform,
                SelectState("crop_params", "pad"),
            ),
        )

        self.surface_translation = SubPipeline(
            PipelineModule(
                TranslationTransform,
                SelectState("crop_params", "offset"),
                invert=True,
                device=self.device,
            ),
            CheckCoordsInside(self.config.out_size, device=self.device),
        )

        self.affine_adjuster = SubPipeline(
            PipelineModule(
                AdjustAffineToSpatialCrop,
                SelectState("crop_params", "offset"),
                device=self.device,
            ),
        )

    def build_image(self):
        return Pipeline(
            # Synthesize image
            SelectImage(self.config.generation_image),
            PipelineModule(
                SynthesizeIntensityImage,
                mu_low=25.0,
                mu_high=200.0,
                sigma_global_scale=4.0,
                sigma_local_scale=2.5,
                # sigma_global_scale=6.0,  # 0.5 mm
                # sigma_local_scale=5.0,  # 0.5 mm
                pv_sigma_range=[0.5, 1.2],
                pv_tail_length=3.0,
                pv_from_distances=False,
                min_cortical_contrast=10.0,
                photo_mode=SelectState("photo_mode"),
                device=self.device,
            ),
            RandGammaTransform(prob=0.33),
            self.extracerebral_augmentation,
            self.image_crop,  # Transform to output FOV
            PipelineModule(
                RandBiasfield,
                self.config.out_size,
                scale_min=0.01,
                scale_max=0.04,
                std=0.3,  # 3T
                # std=0.6,  # 7T
                photo_mode=SelectState("photo_mode"),
                photo_spacing=SelectState("photo_spacing"),
                prob=0.75,
                device=self.device,
            ),
            self.resolution_augmentation,
            self.intensity_normalization,
        )

    def build_images(self):
        return Pipelines(
            t1w=Pipeline(
                SelectImage("t1w"),
                self.extracerebral_augmentation,
                self.image_crop,
                self.intensity_normalization,
                skip_on_InputSelectorError=True,
            ),
            # t1w_mask=Pipeline(
            #     SelectImage("t1w_mask"),
            #     self.mask_hemi,
            #     self.image_crop,
            #     skip_on_InputSelectorError=True,
            # ),
            # brain_dist_map=Pipeline(
            #     SelectImage("lp_dist_map", "rp_dist_map", mode="first"),
            #     self.image_crop,
            #     skip_on_InputSelectorError=True,
            # ),
        )

    def build_output(self):
        return OutputPipelines(
            image=self.build_image(),
            affine=Pipeline(
                SelectAffine(self.config.generation_image), self.affine_adjuster
            ),
            # images=self.build_images(),
        )


class ExvivoCropSelect(ExvivoCropSynth):
    def build_state(self):
        state = super().build_state()

        state["selectable_images"] = Pipeline(
            SelectState("available_images"),
            Intersection(self.config.selectable_images),
            unpack_inputs=False,
        )
        return state

    def build_image(self):
        return Pipeline(
            # Select one of the available (valid) images (equal probability of
            # selecting each image)
            PipelineModule(
                SelectImage,
                PipelineModule(
                    RandomChoice,
                    SelectState("selectable_images"),
                ),
            ),
            self.extracerebral_augmentation,
            self.image_crop,  # Transform to output FOV
            self.resolution_augmentation,
            self.intensity_normalization,
        )

    def build_output(self):
        return OutputPipelines(
            image=self.build_image(),
            affine=Pipeline(
                PipelineModule(
                    SelectAffine,
                    SelectState("available_images", 0),
                ),
                self.affine_adjuster,
            ),
            # images=self.build_images(),
        )


class ExvivoSynthLinearMix(ExvivoSynth):
    """Similar to SingleHemiSynth except that it includes a step which linearly
    combines the synthesized image with one or all of the available (real)
    images.
    """

    def build_state(self):
        state = super().build_state()

        state["selectable_images"] = Pipeline(
            SelectState("available_images"),
            Intersection(self.config.selectable_images),
            unpack_inputs=False,
        )
        state["selectable_tensors"] = Pipeline(
            # Get selectable_images for linear combination
            PipelineModule(
                SelectImage,
                packed_args=SelectState("selectable_images"),
                mode="all valid",
            )
        )
        return state

    def build_intensity_transforms(self, *args, **kwargs):
        super().build_intensity_transforms(*args, **kwargs)
        self.combine_images = SubPipeline(
            PipelineModule(
                RandCombineImages,
                SelectState("selectable_tensors"),
                mode="random",
                prob=0.5,
            )
        )

    def build_image(self):
        return Pipeline(
            # Synthesize image
            SelectImage(self.config.generation_image),
            PipelineModule(
                SynthesizeIntensityImage,
                mu_low=25.0,
                mu_high=200.0,
                sigma_global_scale=4.0,
                sigma_local_scale=2.5,
                # sigma_global_scale=6.0,  # 0.5 mm
                # sigma_local_scale=5.0,  # 0.5 mm
                pv_sigma_range=[0.5, 1.2],
                pv_tail_length=3.0,
                pv_from_distances=False,
                min_cortical_contrast=10.0,
                photo_mode=SelectState("photo_mode"),
                device=self.device,
            ),
            self.combine_images,
            RandGammaTransform(prob=0.33),
            self.mask_hemi,
            self.image_deformation,  # Transform to output FOV
            PipelineModule(
                RandBiasfield,
                self.config.out_size,
                scale_min=0.01,
                scale_max=0.04,
                std=0.3,  # 3T
                # std=0.6,  # 7T
                photo_mode=SelectState("photo_mode"),
                photo_spacing=SelectState("photo_spacing"),
                prob=0.75,
                device=self.device,
            ),
            self.resolution_augmentation,
            self.intensity_normalization,
        )
