import torch


from brainsynth.transforms import *


mapped_inputs = dict(
    image=dict(
        T1w=torch.rand((10,10,10)),
        T2w=10*torch.rand((10,10,10)),
        generation=torch.randint(0, 20, (128,192,176)),
        # FLAIR=torch.rand((10,10,10))
    ),
    surface=dict(lh=dict(white=3, pial=4)),
    initial_vertices={},
    state=dict(
        in_res = torch.tensor([1,1,1]),
        out_res = torch.tensor([1,1,1]),
        center=torch.tensor([3,4,5]),
        available_contrasts=("image:T1w", "image:T2w", "image:FLAIR"),
        alternative_contrasts=("image:T1w", "image:T2w", "image:FLAIR", "image:CT")
    )
)

mytrans = PipelineModule(transformation, args, kwargs)
mytrans1 = PipelineModule(transformation1, args, kwargs)
mytrans2 = PipelineModule(transformation2, args, kwargs)

mypipeline = Pipeline(mytrans, mytrans1, mytrans2)




device = torch.device("cuda")
out_size = (128, 192, 176)
out_center = "lh"


RandSkullStrip = RandMaskFromLabelImageRemove(
    InputSelector("image:segmentation"), labels = (1,2,3,4),
)

OneHotEncode = AsDiscreteWithReindex(labels)

# InputSelector is special: its specification below concerns its call and not
# initialization like the other classes. InputSelector is initialized once at
# the start of augmentation.

photo_mode = RandomChoice((False, True))()

spatial_crop = PipelineModule(
    SpatialCrop,
    InputSelector("state:in_size"),
    # "input:state:in_size",
    out_size,
    InputSelector("state:center"),
    # "input:state:center",
)

intensity_normalization = IntensityNormalization()

nonlinear_transform = RandNonlinearTransform(
    out_size,
    scale_min = 0.03,
    scale_max = 0.06,
    std_max = 4.0,
    prob = 0.0,
    device = device,
)
linear_transform = RandLinearTransform(
    max_rotation = 15,
    max_scale = 0.2,
    max_shear = 0.2,
    prob = 0.0,
    device = device,
)
translation_transform =

# Compute state variables

# Initial state variables:
#   in_size     input image size (spatial dimensions)
#   in_res      input image resolution
#   center      center of the output wrt. the input
#   grid        undistorted output grid

STATE = dict(
    resampling_grid = Pipeline(
        InputSelector("state:grid"),
        CenterGrid(),
        nonlinear_transform,
        linear_transform,
        PipelineModule(
            TranslationTransform,
            InputSelector("state:center"),
        )
    )
)

GridSample(InputSelector("state:resampling_grid"))

"""
Pipeline provides `mapped_inputs` which are needed when calling InputSelector.
Hence, all InputSelectors must be used *in* a Pipeline.

PipelineModule defers class initialization until runtime and provides `mapped_inputs`
at class instatiation. Consequently, all PipelineModules must be used *in* a
Pipeline.
By deferring initialization until runtime, PipelineModule allows transforms to be
initialized with arguments whose values are only available at runtime.


"""

# IMAGE PIPELINES

OUTPUT = dict(
    T1w = Pipeline(
        InputSelector("image:T1w"),
        intensity_normalization,
    ),
    T2w = Pipeline(
        InputSelector("image:T2w"),
        intensity_normalization,
    ),
    segmentation = Pipeline(
        InputSelector("image:segmentation"),
        OneHotEncoding(num_classes)
    ),
    synth = Pipeline(
        RandomChoice(
            [
                # Synthesize image
                Pipeline(
                    InputSelector("image:generation"),
                    SynthesizeIntensityImage(
                        mu_offset = 25.0,
                        mu_scale = 200.0,
                        sigma_offset = 0.0,
                        sigma_scale = 10.0,
                        add_partial_volume = True,
                        photo_mode = photo_mode,
                        min_cortical_contrast = 25.0,
                        device = device,
                    ),
                    # RandGaussianNoise(),
                    RandGammaTransform(
                        mean = 0.0, std = 1.0, prob = 0.75
                    ),
                    RandBiasfield(
                        out_size,
                        scale_min = 0.02,
                        scale_max = 0.04,
                        std_min = 0.1,
                        std_max = 0.6,
                        photo_mode = photo_mode,
                        prob = 0.9,
                    ),
                ),
                # Select actual image
                Pipeline(
                    # InputSelector(
                    #     RandomChoice(["image:T1w", "image:T2w"], prob=(0.5, 0.5))(),
                    # )
                    # select available contrasts...
                    PipelineModule(
                        InputSelector,
                        PipelineModule(
                            RandomChoice,
                            InputSelector("state:available_contrasts"),
                        ),
                    ),
                )
            ], prob=(0.5, 0.5)
        ),
        # Resolution
        PipelineModule(
            RandResolution,
            out_size,
            InputSelector("state:in_res"),
            InputSelector("state:out_res"),
            prob = 0.75,
        ),
        intensity_normalization,
    ),


    surfaces = Pipeline(
        InputSelector("surface"),
        # ...
    ),
)




p = Pipeline(
    # PipelineModule(
    #     "InputSelector",
    #     RandomChoice("image:T1w", "image:T2w", prob=(0.5, 0.5))(),
    # ),
    PipelineModule(
        "InputSelector",
        PipelineModule(
            RandomChoice,
            "input:state:available_contrasts",
            prob=(0.5, 0.3, 0.2)
        ),
    ),
    NormalizeIntensity(0.1, 0.9),
    PipelineModule(
        SpatialCrop,
        in_size=torch.tensor([10,10,10]),
        out_size="input:state:fov",
        out_center="input:state:center"
    )
)
p.initialize(inp_sel)
p()






T1w = SequentialTransform("input:image:T1w", normintensity)
distances = SequentialTransform("input:image:distblabla", Scaler("input:state:scale"))



out = {}
for pipeline in pipelines:
    trans = pipeline[0](inp_sel)
    res = trans()
    for transform in pipeline[1:]:
        res = transform(inp_sel)(res)


pipelines = OrderedDict(



    T1mask = [
        InputSelector("image:T1-mask"),
        skullstrip,
    ],

    distances = [
        InputSelector("image:distances"),
        Scaler(InputSelector("state:scale")),
    ],



    initial_vertices = SequentialTransform(
        InputSelector("initial_vertices"),
    ),

)
