from functools import partial

from brainsynth.constants.constants import mapped_input_keys as mikeys
from brainsynth.config.augmentation_configuration import AugmentationConfiguration
from brainsynth.transforms import *

# Use from_image("T1w") instead of "image:T1w".
from_image = lambda x: f"{mikeys.image}:{x}"
from_initial_vertices = lambda x: f"{mikeys.initial_vertices}:{x}"
from_surface = lambda x: f"{mikeys.surface}:{x}"
from_state = lambda x: f"{mikeys.state}:{x}"


class PipelineBuilder:
    def __init__(self) -> None:
        pass

    def build(self, config: AugmentationConfiguration):
        raise NotImplementedError()


class DefaultPipeline(PipelineBuilder):
    def __init__(self) -> None:
        super().__init__()

    def build(self, config: AugmentationConfiguration):
        """
        Pipeline provides `mapped_inputs` which are needed when calling InputSelector.
        Hence, all InputSelectors must be used *in* a Pipeline.

        PipelineModule defers class initialization until runtime and provides `mapped_inputs`
        at class instatiation. Consequently, all PipelineModules must be used *in* a
        Pipeline.
        By deferring initialization until runtime, PipelineModule allows transforms to be
        initialized with arguments whose values are only available at runtime.


        Returns
        -------
        state :
            A dictionary of Pipelines/transformations that updates the state of
            the synthesizer each time it is called
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
            backward_field_estimation=True,
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
            in_size=Pipeline(
                SelectImage(
                    "generation"
                ),  # this should be an image that is always available
                SpatialSize(),
            ),
            #
            surface_bbox=Pipeline(
                SelectInitialVertices(),
                # InputSelector("surface:lh:pial"),
                SurfaceBoundingBox(),
                unpack_inputs = False,
            ),
            # center of FOV (output) in the input image(s)
            out_center=Pipeline(
                ServeValue(config.out_center_str),
                PipelineModule(
                    CenterFromString,
                    SelectState("in_size"),
                    SelectState("surface_bbox"),
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
            CheckCoordsInside(config.out_size),
        )

        output = dict(
            T1w=Pipeline(
                SelectImage("T1w"),
                image_deformation,
                intensity_normalization,
            ),
            T1w_mask=Pipeline(
                SelectImage("T1w_mask"),
                image_deformation,
            ),
            T2w=Pipeline(
                SelectImage("T2w"),
                image_deformation,
                intensity_normalization,
            ),
            T2w_mask=Pipeline(
                SelectImage("T2w_mask"),
                image_deformation,
            ),
            segmentation=Pipeline(
                SelectImage("segmentation"),
                image_deformation,
                OneHotEncoding(config.segmentation_num_labels),
            ),
            image=Pipeline(
                RandomChoice(
                    [
                        # Synthesize image
                        Pipeline(
                            SelectImage("generation"),
                            SynthesizeIntensityImage(
                                mu_offset=25.0,
                                mu_scale=200.0,
                                sigma_offset=0.0,
                                sigma_scale=10.0,
                                add_partial_volume=True,
                                photo_mode=config.photo_mode,
                                min_cortical_contrast=25.0,
                                device=device,
                            ),
                            # RandGaussianNoise(),
                            RandGammaTransform(mean=0.0, std=1.0, prob=0.75),
                        ),
                        # Select actual image
                        Pipeline(
                            # SelectImage(
                            #     RandomChoice(["image:T1w", "image:T2w"], prob=(0.5, 0.5))(),
                            # )
                            # select one of the available contrasts...
                            PipelineModule(
                                SelectImage,
                                PipelineModule(
                                    RandomChoice,
                                    SelectState("available_images"),
                                ),
                            ),
                        ),
                    ],
                    prob=(0.5, 0.5),
                ),
                #
                image_deformation,
                # Biasfield
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
                # Resolution
                PipelineModule(
                    RandResolution,
                    config.out_size,
                    config.in_res,
                    photo_mode=config.photo_mode,
                    photo_spacing=SelectState("photo_spacing"),
                    photo_thickness=config.photo_thickness,
                    prob=0.75,
                    device=device,
                ),
                intensity_normalization,
            ),
            surface=Pipeline(
                SelectSurface(),
                surface_deformation,
            ),
            initial_vertices=Pipeline(
                SelectInitialVertices(),
                surface_deformation,
            ),
        )

        return state, output
