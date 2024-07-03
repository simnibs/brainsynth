from functools import partial

from brainsynth.constants import mapped_input_keys as mikeys
from brainsynth.config.synthesizer import SynthesizerConfig
from brainsynth.transforms import *


class SynthBuilder:
    def __init__(self) -> None:
        pass

    def build(self, config: SynthesizerConfig):
        raise NotImplementedError()


class DefaultSynthBuilder(SynthBuilder):
    def __init__(self) -> None:
        """This pipeline includes

            - linear deformation
            - nonlinear deformation
            - synthesis of an image as the input image for training

        """
        super().__init__()

    def build(self, config: SynthesizerConfig):
        """Build a full synthesizer pipeline by collecting multiple Pipeline
        instances.

        Parameters
        ----------
        config : SynthesizerConfig


        Returns
        -------
        state :
            A dictionary of Pipelines/transformations that updates the state of
            the synthesizer each time it is called.
        output :
            A dictionary of Pipelines/transformations that generates the
            outputs, e.g., a synthesized image and processed T1, T2, surfaces,
            etc.
        """
        device = config.device

        # ---------------------------------------------------------------------
        # The spatial augmentation pipeline
        # ---------------------------------------------------------------------

        nonlinear_transform = RandNonlinearTransform(
            config.out_size,
            scale_min=0.03,
            scale_max=0.06,
            std_max=4.0,
            exponentiate_field=True,
            grid=config.grid,
            prob=1.0,
            device=device,
        )
        linear_transform = RandLinearTransform(
            max_rotation=15.0,
            max_scale=0.2,
            max_shear=0.2,
            prob=1.0,
            device=device,
        )

        inv_nonlinear_transform = partial(nonlinear_transform, direction="backward")
        inv_linear_transform = partial(linear_transform, inverse=True)

        state = dict(
            # spacing of photos (slices)
            photo_spacing=Uniform(*config.photo_spacing_range, device=device),
            # the size of the input image(s)
            # (this should be an image that is always available)
            in_size=Pipeline(
                SelectImage("generation_labels"),
                SpatialSize(),
            ),
            # SurfaceBoundingBox needs some work
            surface_bbox=Pipeline(
                SelectInitialVertices(),
                SurfaceBoundingBox(),
                unpack_inputs=False,
            ),
            # where to center the (output) FOV in the input image space
            out_center=Pipeline(
                ServeValue(config.out_center_str),
                PipelineModule(
                    CenterFromString,
                    SelectState("in_size"),
                    SelectState("surface_bbox"),
                    align_corners=config.align_corners,
                ),
            ),
            # the deformed grid
            resampling_grid=Pipeline(
                ServeValue(config.centered_grid),
                nonlinear_transform,
                linear_transform,
                PipelineModule(
                    TranslationTransform,
                    SelectState("out_center"),
                    device=device,
                ),
            ),
        )

        # ---------------------------------------------------------------------
        # The output generation pipeline
        # ---------------------------------------------------------------------

        # Example 1: Random skull-stripping

        # RandSkullStrip = PipelineModule(
        #     RandMaskRemove,
        #     PipelineModule(
        #         MaskFromLabelImage,
        #         SelectImage("segmentation"),
        #         labels = (0,),
        #         device = device,
        #     ),
        # )

        intensity_normalization = IntensityNormalization()

        image_deformation = SubPipeline(
            PipelineModule(
                GridSample,
                SelectState("resampling_grid"),
                SelectState("in_size"),
            ),
        )

        surface_deformation = SubPipeline(
            PipelineModule(
                TranslationTransform,
                SelectState("out_center"),
                invert=True,
                device=device,
            ),
            inv_linear_transform,
            TranslationTransform(config.center, device=device),
            inv_nonlinear_transform,
            CheckCoordsInside(config.out_size, device=device),
        )

        output = dict(
            t1w=Pipeline(
                SelectImage("t1w"),
                image_deformation,
                intensity_normalization,
                # This pipeline is allowed to fail if the input is not found
                skip_on_InputSelectorError=True,
            ),
            t1w_mask=Pipeline(
                SelectImage("t1w_mask"),
                image_deformation,
                skip_on_InputSelectorError=True,
            ),
            t2w=Pipeline(
                SelectImage("t2w"),
                image_deformation,
                intensity_normalization,
                skip_on_InputSelectorError=True,
            ),
            t2w_mask=Pipeline(
                SelectImage("t2w_mask"),
                image_deformation,
                skip_on_InputSelectorError=True,
            ),
            brainseg=Pipeline(
                SelectImage("brainseg"),
                image_deformation,
                OneHotEncoding(config.segmentation_num_labels),
                skip_on_InputSelectorError=True,
            ),
            # Synthesize image or select an actual image
            image=Pipeline(
                SelectImage("generation_labels"),
                SynthesizeIntensityImage(
                    mu_offset=25.0,
                    mu_scale=200.0,
                    sigma_offset=0.0,
                    sigma_scale=10.0,
                    photo_mode=config.photo_mode,
                    min_cortical_contrast=25.0,
                    device=device,
                ),
                # RandGaussianNoise(),
                RandGammaTransform(mean=0.0, std=1.0, prob=0.75),
                image_deformation,
                RandBiasfield(
                    config.out_size,
                    scale_min=0.02,
                    scale_max=0.04,
                    std_min=0.1,
                    std_max=0.6,
                    photo_mode=config.photo_mode,
                    prob=0.9,
                    device=device,
                ),
                PipelineModule(
                    RandResolution,
                    config.out_size,
                    config.in_res,
                    photo_mode=config.photo_mode,
                    photo_spacing=SelectState("photo_spacing"),
                    photo_thickness=config.photo_thickness,
                    prob=0.0,
                    device=device,
                ),
                intensity_normalization,
            ),
            surface=Pipeline(
                SelectSurface(),
                surface_deformation,
                skip_on_InputSelectorError=True,
            ),
            initial_vertices=Pipeline(
                SelectInitialVertices(),
                surface_deformation,
                skip_on_InputSelectorError=True,
            ),
        )

        return state, output


class OnlySynthBuilder(SynthBuilder):
    def __init__(self) -> None:
        """This pipeline includes

            - synthesis of an image as the input image for training

        and thus *no* spatial augmentation (only extracting a particular FOV).
        """
        super().__init__()

    def build(self, config: SynthesizerConfig):
        """Build a full synthesizer pipeline by collecting multiple Pipeline
        instances.

        Parameters
        ----------
        config : SynthesizerConfig


        Returns
        -------
        state :
            A dictionary of Pipelines/transformations that updates the state of
            the synthesizer each time it is called.
        output :
            A dictionary of Pipelines/transformations that generates the
            outputs, e.g., a synthesized image and processed T1, T2, surfaces,
            etc.
        """
        device = config.device

        # ---------------------------------------------------------------------
        # The spatial augmentation pipeline
        # ---------------------------------------------------------------------

        state = dict(
            # the size of the input image(s)
            # (this should be an image that is always available)
            in_size=Pipeline(
                SelectImage("generation_labels"),
                SpatialSize(),
            ),
            # SurfaceBoundingBox needs some work
            surface_bbox=Pipeline(
                SelectInitialVertices(),
                SurfaceBoundingBox(),
                unpack_inputs=False,
            ),
            # where to center the (output) FOV in the input image space
            out_center=Pipeline(
                ServeValue(config.out_center_str),
                PipelineModule(
                    CenterFromString,
                    SelectState("in_size"),
                    SelectState("surface_bbox"),
                    align_corners=config.align_corners,
                ),
            ),
            # the deformed grid
            resampling_grid=Pipeline(
                ServeValue(config.centered_grid),
                PipelineModule(
                    TranslationTransform,
                    SelectState("out_center"),
                    device=device,
                ),
            ),
        )

        # ---------------------------------------------------------------------
        # The output generation pipeline
        # ---------------------------------------------------------------------

        intensity_normalization = IntensityNormalization()

        image_deformation = SubPipeline(
            PipelineModule(
                GridSample,
                SelectState("resampling_grid"),
                SelectState("in_size"),
            ),
        )

        surface_deformation = SubPipeline(
            PipelineModule(
                TranslationTransform,
                SelectState("out_center"),
                invert=True,
                device=device,
            ),
            TranslationTransform(config.center, device=device),
            CheckCoordsInside(config.out_size, device=device),
        )

        output = dict(
            t1w=Pipeline(
                SelectImage("t1w"),
                image_deformation,
                intensity_normalization,
                # This pipeline is allowed to fail if the input is not found
                skip_on_InputSelectorError=True,
            ),
            t1w_mask=Pipeline(
                SelectImage("t1w_mask"),
                image_deformation,
                skip_on_InputSelectorError=True,
            ),
            t2w=Pipeline(
                SelectImage("t2w"),
                image_deformation,
                intensity_normalization,
                skip_on_InputSelectorError=True,
            ),
            t2w_mask=Pipeline(
                SelectImage("t2w_mask"),
                image_deformation,
                skip_on_InputSelectorError=True,
            ),
            brainseg=Pipeline(
                SelectImage("brainseg"),
                image_deformation,
                OneHotEncoding(config.segmentation_num_labels),
                skip_on_InputSelectorError=True,
            ),
            # Synthesize image or select an actual image
            image=Pipeline(
                SelectImage("generation_labels"),
                SynthesizeIntensityImage(
                    mu_offset=25.0,
                    mu_scale=200.0,
                    sigma_offset=0.0,
                    sigma_scale=10.0,
                    photo_mode=config.photo_mode,
                    min_cortical_contrast=25.0,
                    device=device,
                ),
                # RandGaussianNoise(),
                RandGammaTransform(mean=0.0, std=1.0, prob=0.75),
                image_deformation,
                RandBiasfield(
                    config.out_size,
                    scale_min=0.02,
                    scale_max=0.04,
                    std_min=0.1,
                    std_max=0.6,
                    photo_mode=config.photo_mode,
                    prob=0.9,
                    device=device,
                ),
                PipelineModule(
                    RandResolution,
                    config.out_size,
                    config.in_res,
                    prob=0.0,
                    device=device,
                ),
                intensity_normalization,
            ),
            surface=Pipeline(
                SelectSurface(),
                surface_deformation,
                skip_on_InputSelectorError=True,
            ),
            initial_vertices=Pipeline(
                SelectInitialVertices(),
                surface_deformation,
                skip_on_InputSelectorError=True,
            ),
        )

        return state, output



class OnlySelectBuilder(SynthBuilder):
    def __init__(self) -> None:
        """This pipeline includes

            - selection of an image as the input image for training

        and thus *no* spatial augmentation (only extracting a particular FOV).
        """
        super().__init__()

    def build(self, config: SynthesizerConfig):
        """Build a full synthesizer pipeline by collecting multiple Pipeline
        instances.

        Parameters
        ----------
        config : SynthesizerConfig


        Returns
        -------
        state :
            A dictionary of Pipelines/transformations that updates the state of
            the synthesizer each time it is called.
        output :
            A dictionary of Pipelines/transformations that generates the
            outputs, e.g., a synthesized image and processed T1, T2, surfaces,
            etc.
        """
        device = config.device

        # ---------------------------------------------------------------------
        # The spatial augmentation pipeline
        # ---------------------------------------------------------------------

        state = dict(
            # available images
            available_images=Pipeline(
                SelectImage(),
                ExtractDictKeys(),
                Intersection(config.alternative_images),
                unpack_inputs=False,
            ),
            # the size of the input image(s)
            # (this should be an image that is always available)
            in_size=Pipeline(
                PipelineModule(
                    SelectImage,
                    # doesn't matter which is selected - all should be the same size!
                    PipelineModule(
                        RandomChoice,
                        SelectState("available_images"),
                    ),
                ),
                SpatialSize(),
            ),
            # SurfaceBoundingBox needs some work
            surface_bbox=Pipeline(
                SelectInitialVertices(),
                SurfaceBoundingBox(),
                unpack_inputs=False,
            ),
            # where to center the (output) FOV in the input image space
            out_center=Pipeline(
                ServeValue(config.out_center_str),
                PipelineModule(
                    CenterFromString,
                    SelectState("in_size"),
                    SelectState("surface_bbox"),
                    align_corners=config.align_corners,
                ),
            ),
            # the deformed grid
            resampling_grid=Pipeline(
                ServeValue(config.centered_grid),
                PipelineModule(
                    TranslationTransform,
                    SelectState("out_center"),
                    device=device,
                ),
            ),
        )

        # ---------------------------------------------------------------------
        # The output generation pipeline
        # ---------------------------------------------------------------------

        intensity_normalization = IntensityNormalization()

        image_deformation = SubPipeline(
            PipelineModule(
                GridSample,
                SelectState("resampling_grid"),
                SelectState("in_size"),
            ),
        )

        surface_deformation = SubPipeline(
            PipelineModule(
                TranslationTransform,
                SelectState("out_center"),
                invert=True,
                device=device,
            ),
            TranslationTransform(config.center, device=device),
            CheckCoordsInside(config.out_size, device=device),
        )

        output = dict(
            t1w=Pipeline(
                SelectImage("t1w"),
                image_deformation,
                intensity_normalization,
                # This pipeline is allowed to fail if the input is not found
                skip_on_InputSelectorError=True,
            ),
            t1w_mask=Pipeline(
                SelectImage("t1w_mask"),
                image_deformation,
                skip_on_InputSelectorError=True,
            ),
            t2w=Pipeline(
                SelectImage("t2w"),
                image_deformation,
                intensity_normalization,
                skip_on_InputSelectorError=True,
            ),
            t2w_mask=Pipeline(
                SelectImage("t2w_mask"),
                image_deformation,
                skip_on_InputSelectorError=True,
            ),
            brainseg=Pipeline(
                SelectImage("brainseg"),
                image_deformation,
                OneHotEncoding(config.segmentation_num_labels),
                skip_on_InputSelectorError=True,
            ),
            # Synthesize image or select an actual image
            image=Pipeline(
                # select one of the available contrasts...
                PipelineModule(
                    SelectImage,
                    PipelineModule(
                        RandomChoice,
                        SelectState("available_images"),
                        # equal probability of selecting each image
                    ),
                ),
                image_deformation,
                PipelineModule(
                    RandResolution,
                    config.out_size,
                    config.in_res,
                    prob=0.0,
                    device=device,
                ),
                intensity_normalization,
            ),
            surface=Pipeline(
                SelectSurface(),
                surface_deformation,
                skip_on_InputSelectorError=True,
            ),
            initial_vertices=Pipeline(
                SelectInitialVertices(),
                surface_deformation,
                skip_on_InputSelectorError=True,
            ),
        )

        return state, output


class SynthOrSelectImageBuilder(SynthBuilder):
    def __init__(self) -> None:
        """This pipeline includes

            - linear deformation
            - nonlinear deformation
            - A choice between a real input image or the generation of a
              synthetic image as the input image for training

        """
        super().__init__()

    def build(self, config: SynthesizerConfig):
        """Build a full synthesizer pipeline by collecting multiple Pipeline
        instances.

        Parameters
        ----------
        config : SynthesizerConfig


        Returns
        -------
        state :
            A dictionary of Pipelines/transformations that updates the state of
            the synthesizer each time it is called.
        output :
            A dictionary of Pipelines/transformations that generates the
            outputs, e.g., a synthesized image and processed T1, T2, surfaces,
            etc.
        """
        device = config.device

        # ---------------------------------------------------------------------
        # The spatial augmentation pipeline
        # ---------------------------------------------------------------------

        nonlinear_transform = RandNonlinearTransform(
            config.out_size,
            scale_min=0.03,
            scale_max=0.06,
            std_max=4.0,
            exponentiate_field=True,
            grid=config.grid,
            prob=1.0,
            device=device,
        )
        linear_transform = RandLinearTransform(
            max_rotation=15.0,
            max_scale=0.2,
            max_shear=0.2,
            prob=1.0,
            device=device,
        )

        inv_nonlinear_transform = partial(nonlinear_transform, direction="backward")
        inv_linear_transform = partial(linear_transform, inverse=True)

        state = dict(
            # spacing of photos (slices)
            photo_spacing=Uniform(*config.photo_spacing_range, device=device),
            # available images
            available_images=Pipeline(
                SelectImage(),
                ExtractDictKeys(),
                Intersection(config.alternative_images),
                unpack_inputs=False,
            ),
            # the size of the input image(s)
            # (this should be an image that is always available)
            in_size=Pipeline(
                SelectImage("generation_labels"),
                SpatialSize(),
            ),
            # SurfaceBoundingBox needs some work
            surface_bbox=Pipeline(
                SelectInitialVertices(),
                SurfaceBoundingBox(),
                unpack_inputs=False,
            ),
            # where to center the (output) FOV in the input image space
            out_center=Pipeline(
                ServeValue(config.out_center_str),
                PipelineModule(
                    CenterFromString,
                    SelectState("in_size"),
                    SelectState("surface_bbox"),
                    align_corners=config.align_corners,
                ),
            ),
            # the deformed grid
            resampling_grid=Pipeline(
                ServeValue(config.centered_grid),
                nonlinear_transform,
                linear_transform,
                PipelineModule(
                    TranslationTransform,
                    SelectState("out_center"),
                    device=device,
                ),
            ),
        )

        # ---------------------------------------------------------------------
        # The output generation pipeline
        # ---------------------------------------------------------------------

        # Example 1: Random skull-stripping

        # RandSkullStrip = PipelineModule(
        #     RandMaskRemove,
        #     PipelineModule(
        #         MaskFromLabelImage,
        #         SelectImage("segmentation"),
        #         labels = (0,),
        #         device = device,
        #     ),
        # )

        intensity_normalization = IntensityNormalization()

        image_deformation = SubPipeline(
            PipelineModule(
                GridSample,
                SelectState("resampling_grid"),
                SelectState("in_size"),
            ),
        )

        surface_deformation = SubPipeline(
            PipelineModule(
                TranslationTransform,
                SelectState("out_center"),
                invert=True,
                device=device,
            ),
            inv_linear_transform,
            TranslationTransform(config.center, device=device),
            inv_nonlinear_transform,
            CheckCoordsInside(config.out_size, device=device),
        )

        output = dict(
            t1w=Pipeline(
                SelectImage("t1w"),
                image_deformation,
                intensity_normalization,
                # This pipeline is allowed to fail if the input is not found
                skip_on_InputSelectorError=True,
            ),
            t1w_mask=Pipeline(
                SelectImage("t1w_mask"),
                image_deformation,
                skip_on_InputSelectorError=True,
            ),
            t2w=Pipeline(
                SelectImage("t2w"),
                image_deformation,
                intensity_normalization,
                skip_on_InputSelectorError=True,
            ),
            t2w_mask=Pipeline(
                SelectImage("t2w_mask"),
                image_deformation,
                skip_on_InputSelectorError=True,
            ),
            brainseg=Pipeline(
                SelectImage("brainseg"),
                image_deformation,
                OneHotEncoding(config.segmentation_num_labels),
                skip_on_InputSelectorError=True,
            ),
            # Synthesize image or select an actual image
            image=Pipeline(
                RandomChoice(
                    [
                        Pipeline(
                            SelectImage("generation_labels"),
                            SynthesizeIntensityImage(
                                mu_offset=25.0,
                                mu_scale=200.0,
                                sigma_offset=0.0,
                                sigma_scale=10.0,
                                photo_mode=config.photo_mode,
                                min_cortical_contrast=25.0,
                                device=device,
                            ),
                            # RandGaussianNoise(),
                            RandGammaTransform(mean=0.0, std=1.0, prob=0.75),
                        ),
                        Pipeline(
                            # select one of the available contrasts...
                            PipelineModule(
                                SelectImage,
                                PipelineModule(
                                    RandomChoice,
                                    SelectState("available_images"),
                                    # equal probability of selecting each image
                                ),
                            ),
                        ),
                    ],
                    prob=(0.9, 0.1),
                ),
                image_deformation,
                RandBiasfield(
                    config.out_size,
                    scale_min=0.02,
                    scale_max=0.04,
                    std_min=0.1,
                    std_max=0.6,
                    photo_mode=config.photo_mode,
                    prob=0.9,
                    device=device,
                ),
                PipelineModule(
                    RandResolution,
                    config.out_size,
                    config.in_res,
                    photo_mode=config.photo_mode,
                    photo_spacing=SelectState("photo_spacing"),
                    photo_thickness=config.photo_thickness,
                    prob=0.0,
                    device=device,
                ),
                intensity_normalization,
            ),
            surface=Pipeline(
                SelectSurface(),
                surface_deformation,
                skip_on_InputSelectorError=True,
            ),
            initial_vertices=Pipeline(
                SelectInitialVertices(),
                surface_deformation,
                skip_on_InputSelectorError=True,
            ),
        )

        return state, output

class DefaultValidationBuilder(SynthBuilder):
    def __init__(self) -> None:
        """This pipline includes

        * linear deformation
        * nonlinear deformation
        * selection of a real image as the input image for training

        """
        super().__init__()

    def build(self, config: SynthesizerConfig):
        """Build a full synthesizer pipeline by collecting multiple Pipeline
        instances.

        Parameters
        ----------
        config : SynthesizerConfig


        Returns
        -------
        state :
            A dictionary of Pipelines/transformations that updates the state of
            the synthesizer each time it is called.
        output :
            A dictionary of Pipelines/transformations that generates the
            outputs, e.g., a synthesized image and processed T1, T2, surfaces,
            etc.
        """
        device = config.device

        # ---------------------------------------------------------------------
        # The spatial augmentation pipeline
        # ---------------------------------------------------------------------

        nonlinear_transform = RandNonlinearTransform(
            config.out_size,
            scale_min=0.03,
            scale_max=0.06,
            std_max=4.0,
            exponentiate_field=True,
            grid=config.grid,
            prob=1.0,
            device=device,
        )
        linear_transform = RandLinearTransform(
            max_rotation=15.0,
            max_scale=0.2,
            max_shear=0.2,
            prob=1.0,
            device=device,
        )

        inv_nonlinear_transform = partial(nonlinear_transform, direction="backward")
        inv_linear_transform = partial(linear_transform, inverse=True)

        state = dict(
            # spacing of photos (slices)
            photo_spacing=Uniform(*config.photo_spacing_range, device=device),
            # available images
            available_images=Pipeline(
                SelectImage(),
                ExtractDictKeys(),
                Intersection(config.alternative_images),
                unpack_inputs=False,
            ),
            # the size of the input image(s)
            # (this should be an image that is always available)
            in_size=Pipeline(
                PipelineModule(
                    SelectImage,
                    # doesn't matter which is selected - all should be the same size!
                    PipelineModule(
                        RandomChoice,
                        SelectState("available_images"),
                    ),
                ),
                SpatialSize(),
            ),
            # SurfaceBoundingBox needs some work
            surface_bbox=Pipeline(
                SelectInitialVertices(),
                SurfaceBoundingBox(),
                unpack_inputs=False,
            ),
            # where to center the (output) FOV in the input image space
            out_center=Pipeline(
                ServeValue(config.out_center_str),
                PipelineModule(
                    CenterFromString,
                    SelectState("in_size"),
                    SelectState("surface_bbox"),
                    align_corners=config.align_corners,
                ),
            ),
            # the deformed grid
            resampling_grid=Pipeline(
                ServeValue(config.centered_grid),
                nonlinear_transform,
                linear_transform,
                PipelineModule(
                    TranslationTransform,
                    SelectState("out_center"),
                    device=device,
                ),
            ),
        )

        # ---------------------------------------------------------------------
        # The output generation pipeline
        # ---------------------------------------------------------------------

        intensity_normalization = IntensityNormalization()

        image_deformation = SubPipeline(
            PipelineModule(
                GridSample,
                SelectState("resampling_grid"),
                SelectState("in_size"),
            ),
        )

        surface_deformation = SubPipeline(
            PipelineModule(
                TranslationTransform,
                SelectState("out_center"),
                invert=True,
                device=device,
            ),
            inv_linear_transform,
            TranslationTransform(config.center, device=device),
            inv_nonlinear_transform,
            CheckCoordsInside(config.out_size, device=device),
        )

        output = dict(
            t1w=Pipeline(
                SelectImage("t1w"),
                image_deformation,
                intensity_normalization,
                # This pipeline is allowed to fail if the input is not found
                skip_on_InputSelectorError=True,
            ),
            t1w_mask=Pipeline(
                SelectImage("t1w_mask"),
                image_deformation,
                skip_on_InputSelectorError=True,
            ),
            t2w=Pipeline(
                SelectImage("t2w"),
                image_deformation,
                intensity_normalization,
                skip_on_InputSelectorError=True,
            ),
            t2w_mask=Pipeline(
                SelectImage("t2w_mask"),
                image_deformation,
                skip_on_InputSelectorError=True,
            ),
            brainseg=Pipeline(
                SelectImage("brainseg"),
                image_deformation,
                OneHotEncoding(config.segmentation_num_labels),
                skip_on_InputSelectorError=True,
            ),
            # Synthesize image or select an actual image
            image=Pipeline(
                # select one of the available contrasts...
                PipelineModule(
                    SelectImage,
                    PipelineModule(
                        RandomChoice,
                        SelectState("available_images"),
                        # equal probability of selecting each image
                    ),
                ),
                image_deformation,
                PipelineModule(
                    RandResolution,
                    config.out_size,
                    config.in_res,
                    photo_mode=config.photo_mode,
                    photo_spacing=SelectState("photo_spacing"),
                    photo_thickness=config.photo_thickness,
                    prob=0.0,
                    device=device,
                ),
                intensity_normalization,
            ),
            surface=Pipeline(
                SelectSurface(),
                surface_deformation,
                skip_on_InputSelectorError=True,
            ),
            initial_vertices=Pipeline(
                SelectInitialVertices(),
                surface_deformation,
                skip_on_InputSelectorError=True,
            ),
        )

        return state, output


class XSubSynthBuilder(SynthBuilder):
    def __init__(self) -> None:
        super().__init__()

    def build(self, config: SynthesizerConfig):
        device = config.device

        # assert (
        #     config.out_center_str == "image"
        # ), "Currently, only `image` center specification is compatible with xsub builder"

        # ---------------------------------------------------------------------
        # The spatial augmentation pipeline
        # ---------------------------------------------------------------------

        warp_surface = SubPipeline(
            PipelineModule(
                XSubWarpSurface_v2, # or XSubWarpSurface_v1
                # SelectState("self_mni_backward_cropped"),
                SelectImage("mni152_nonlin_backward"),
                SelectImage("other_mni152_nonlin_forward"),
            ),
        )

        state = dict(
            # spacing of photos (slices)
            photo_spacing=Uniform(*config.photo_spacing_range, device=device),
            # available images
            available_images=Pipeline(
                SelectImage(),
                ExtractDictKeys(),
                Intersection(config.alternative_images),
                unpack_inputs=False,
            ),
            # the size of the input image(s)
            # (this should be an image that is always available)
            self_size=Pipeline(
                SelectImage("generation_labels"),
                SpatialSize(),
            ),
            # the spatial dimensions of the image to warp *to*
            other_size=Pipeline(
                SelectImage("other_mni152_nonlin_backward"),
                SpatialSize(),
            ),
            surface_bbox=Pipeline(
                SelectInitialVertices(),
                SurfaceBoundingBox(),
                unpack_inputs=False,
            ),
            # padded_surface_bbox=Pipeline(
            #     SelectInitialVertices(),
            #     # pad so hopefully the pial surface is inside...
            #     SurfaceBoundingBox(floor_and_ceil=True, pad=20.0, reduce=True),
            #     PipelineModule(
            #         RestrictBoundingBox,
            #         SelectState("self_size"),
            #         device=device,
            #     ),
            #     unpack_inputs=False,
            # ),
            # where to center the (output) FOV in the input image space
            self_out_center=Pipeline(
                ServeValue(config.out_center_str),
                PipelineModule(
                    CenterFromString,
                    SelectState("self_size"),
                    SelectState("surface_bbox"),
                    align_corners=config.align_corners,
                ),
            ),
            # transform center to `other`
            other_out_center = Pipeline(
                SelectState("self_out_center"),
                ApplyFunction(torch.atleast_2d),
                warp_surface,
                ApplyFunction(torch.squeeze),
            ),
            # other_out_center=Pipeline(
            #     ServeValue(config.out_center_str),
            #     PipelineModule(
            #         CenterFromString,
            #         SelectState("other_size"),
            #         align_corners=config.align_corners,
            #     ),
            # ),
            # cropping parameters in `other` space
            crop_params=Pipeline(
                SelectState("other_size"),
                PipelineModule(
                    SpatialCropParameters,
                    config.out_size,
                    SelectState("other_out_center"),
                ),
            ),
            # we don't need the entire space of `self` in order to transform
            # the surface so crop to a bounding box around the surface(s)
            # self_mni_backward_cropped=Pipeline(
            #     SelectImage("mni152_nonlin_backward"),
            #     PipelineModule(
            #         SpatialCrop,
            #         size=SelectState("self_size"),
            #         start=SelectState("padded_surface_bbox", 0),  # recursive selection
            #         stop=SelectState("padded_surface_bbox", 1),
            #     ),
            # ),
            # depending on the desired output FOV, we don't need to fill the
            # entire space of `other` so crop it
            other_mni_backward_cropped=Pipeline(
                SelectImage("other_mni152_nonlin_backward"),
                PipelineModule(
                    SpatialCrop,
                    size=SelectState("other_size"),
                    slices=SelectState("crop_params", "slices"),
                ),
            ),
        )

        image_deformation = SubPipeline(
            PipelineModule(
                XSubWarpImage,
                SelectImage("mni152_nonlin_forward"),
                SelectState("other_mni_backward_cropped"),
                SelectState("self_size"),  # size of the input we sample *from*
            ),
            # We used a cropped image in `other` space so pad to achieve
            # desired output size
            PipelineModule(
                PadTransform,
                SelectState("crop_params", "pad"),
            ),
        )

        surface_deformation = SubPipeline(
            # compensate for offset caused by cropping in self space
            # PipelineModule(
            #     TranslationTransform,
            #     SelectState("padded_surface_bbox", 0),  # lower left
            #     invert=True,
            #     device=device,
            # ),
            # PipelineModule(
            #     XSubWarpSurface_v1,
            #     SelectState("self_mni_backward_cropped"),
            #     SelectImage("other_mni152_nonlin_forward"),
            # ),
            warp_surface,
            # compensate for offset caused by cropping in other space (because
            # we cropped to out_size)
            PipelineModule(
                TranslationTransform,
                SelectState("crop_params", "offset"),
                invert=True,
                device=device,
            ),
            CheckCoordsInside(config.out_size, device=device),
        )

        # ---------------------------------------------------------------------
        # The output generation pipeline
        # ---------------------------------------------------------------------

        intensity_normalization = IntensityNormalization()

        output = dict(
            t1w=Pipeline(
                SelectImage("t1w"),
                image_deformation,
                intensity_normalization,
                # This pipeline is allowed to fail if the input is not found
                skip_on_InputSelectorError=True,
            ),
            t1w_mask=Pipeline(
                SelectImage("t1w_mask"),
                image_deformation,
                skip_on_InputSelectorError=True,
            ),
            t2w=Pipeline(
                SelectImage("t2w"),
                image_deformation,
                intensity_normalization,
                skip_on_InputSelectorError=True,
            ),
            t2w_mask=Pipeline(
                SelectImage("t2w_mask"),
                image_deformation,
                skip_on_InputSelectorError=True,
            ),
            brainseg=Pipeline(
                SelectImage("brainseg"),
                image_deformation,
                OneHotEncoding(config.segmentation_num_labels),
                skip_on_InputSelectorError=True,
            ),
            # Synthesize image or select an actual image
            image=Pipeline(
                RandomChoice(
                    [
                        Pipeline(
                            SelectImage("generation_labels"),
                            SynthesizeIntensityImage(
                                mu_offset=25.0,
                                mu_scale=200.0,
                                sigma_offset=0.0,
                                sigma_scale=10.0,
                                photo_mode=config.photo_mode,
                                min_cortical_contrast=25.0,
                                device=device,
                            ),
                            # RandGaussianNoise(),
                            RandGammaTransform(mean=0.0, std=1.0, prob=0.75),
                        ),
                        Pipeline(
                            # select one of the available contrasts...
                            PipelineModule(
                                SelectImage,
                                PipelineModule(
                                    RandomChoice,
                                    SelectState("available_images"),
                                    # equal probability of selecting each image
                                ),
                            ),
                        ),
                    ],
                    prob=(1.0, 0.0),
                ),
                image_deformation,
                RandBiasfield(
                    config.out_size,
                    scale_min=0.02,
                    scale_max=0.04,
                    std_min=0.1,
                    std_max=0.6,
                    photo_mode=config.photo_mode,
                    prob=0.9,
                    device=device,
                ),
                PipelineModule(
                    RandResolution,
                    config.out_size,
                    config.in_res,
                    photo_mode=config.photo_mode,
                    photo_spacing=SelectState("photo_spacing"),
                    photo_thickness=config.photo_thickness,
                    prob=0.0,
                    device=device,
                ),
                intensity_normalization,
            ),
            surface=Pipeline(
                SelectSurface(),
                surface_deformation,
                skip_on_InputSelectorError=True,
            ),
            initial_vertices=Pipeline(
                SelectInitialVertices(),
                surface_deformation,
                skip_on_InputSelectorError=True,
            ),
        )

        return state, output
