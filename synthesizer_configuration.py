import torch


from brainsynth.transforms import *

device = torch.device("cuda")


mytrans = Module(transformation, args, kwargs)
mytrans1 = Module(transformation1, args, kwargs)
mytrans2 = Module(transformation2, args, kwargs)

mypipeline = Pipeline(mytrans, mytrans1, mytrans2)

RandSkullStrip = RandMaskFromLabelImageRemove(
    InputSelector("image:segmentation"), labels = (1,2,3,4),
)

OneHotEncode = AsDiscreteWithReindex(labels)

# InputSelector is special: its specification below concerns its call and not
# initialization like the other classes. InputSelector is initialized once at
# the start of augmentation.

photo_mode = RandomChoice((False, True), prob=(0.5, 0.5))()

spatial_crop = Module(
    SpatialCrop,
    "input:state:in_size",
    "input:state:out_size",
    "input:state:center",
)
linear_trans = RandLinearTransform(
    max_rotation = 15,
    max_scale = 0.2,
    max_shear = 0.2,
    prob = 0.0,
    device = device,
)
nonlinear_trans = Module(
    RandNonlinearTransform,
    "input:state:out_size",
    scale_min = 0.03,
    scale_max = 0.06,
    std_max = 4,
    probability = 0.0,
    device = device,
)

intensity_normalization = Module(IntensityNormalization)


pipelines = dict(
    T1w = Pipeline(
        Module(InputSelector, "image:T1w"),
        intensity_normalization,
    ),
    T2w = Pipeline(
        Module(InputSelector, "image:T2w"),
        intensity_normalization,
    ),
    synth = Pipeline(
        RandomChoice(
            [
                Pipeline(
                    Module(InputSelector, "image:generation"),
                    Module(
                        SynthesizeIntensityImage,
                        photo_mode="input:state:photo_mode"
                    ),
                    # RandGaussianNoise(),
                    Module(RandGammaTransform, mean = 0.0, std = 1.0, prob = 0.75),
                    Module(RandBiasfield)
                ),
                Pipeline(
                    Module(
                        InputSelector,
                        RandomChoice(["image:T1w", "image:T2w"], prob=(0.5, 0.5))(),
                    )
                )
            ], prob=(0.5, 0.5)
        ),
        # Resolution
        Module(
            RandResolution,
            "input:state:out_size",
            "input:state:in_res",
            "input:state:out_res",
            prob = 0.75,
        ),
        intensity_normalization,
    )
)


inp_sel = InputSelector(
    images=dict(
        T1w=torch.rand((10,10,10)),
        T2w=10*torch.rand((10,10,10)),
        FLAIR=torch.rand((10,10,10))),
    surfaces=dict(lh=dict(white=3, pial=4)),
    initial_vertices={},
    state=dict(
        fov=torch.tensor([4,5,6]),
        center=torch.tensor([3,4,5]),
        available_contrasts=("image:T1w", "image:T2w", "image:FLAIR"),
    )
)



p = Pipeline(
    Init("InputSelector", "image:T1w"),
    Init(NormalizeIntensity, 0.1, high=0.9),
)
p.initialize(inp_sel)
p()





p = Pipeline(
    # Module(
    #     "InputSelector",
    #     RandomChoice("image:T1w", "image:T2w", prob=(0.5, 0.5))(),
    # ),
    Module(
        "InputSelector",
        Module(
            RandomChoice,
            "input:state:available_contrasts",
            prob=(0.5, 0.3, 0.2)
        ),
    ),
    NormalizeIntensity(0.1, 0.9),
    Module(
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

synth = SequentialTransform(
    Init(
        RandomChoice,
        SequentialTransform(
            Init("input:image:generation"),
            Init(SynthesizeIntensityImage),
            Init(RandGaussianNoise),
            Init(RandGammaTransform),
            Init(RandBiasfield),
        ),
        Init(
            RandomChoice,
            "input:image:T1w",
            "input:image:T2w",
            prob = (0.5, 0.5)),
        prob = (0.9, 0.1),
    ),
    NormalizeIntensity(),
)


out = {}
for pipeline in pipelines:
    trans = pipeline[0](inp_sel)
    res = trans()
    for transform in pipeline[1:]:
        res = transform(inp_sel)(res)


pipelines = OrderedDict(

    synth = [
        RandomChoice(
            SequentialTransform([
                InputSelector("image:generation"),
                SynthesizeIntensityImage(),
                RandGaussianNoise(),
                RandGammaTransform(),
                RandBiasfield(),
            ]),
            RandomChoice("input:image:T1w", "input:image:T2w", prob = (0.5, 0.5)),
            prob = (0.9, 0.1),
        ),
        NormalizeIntensity()
    ],

    T1w = Pipeline(
        "image:T1w",
        NormalizeIntensity(),
        skullstrip,
    ),

    T1mask = [
        InputSelector("image:T1-mask"),
        skullstrip,
    ],

    segmentation = [
        InputSelector("image:segmentation"),
        OneHotEncoding(),
    ],

    distances = [
        InputSelector("image:distances"),
        Scaler(InputSelector("state:scale")),
    ],

    surfaces = SequentialTransform(
        InputSelector("surface"),
        # ...
    ),

    initial_vertices = SequentialTransform(
        InputSelector("initial_vertices"),
    ),

)
