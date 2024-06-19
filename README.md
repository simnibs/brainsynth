# BrainSynth
Brain scan synthesizer.


## Settings

**Basic information** about known types of images, idiosyncracies of how images are encoded (e.g., the way partial volume information is stored, labeling schemes etc.)

    /mrhome/jesperdn/repositories/brainsynth/brainsynth/constants/__init__.py

**Pipelines** are configured in

    /mrhome/jesperdn/repositories/brainsynth/brainsynth/config/synthesizer_builder.py

Dataset configurations for **known datasets** can be generated using `DatasetConfig` from

    /mrhome/jesperdn/repositories/brainsynth/brainsynth/config/dataset.py

## Building Synthesizer Pipelines

Synthesizer instances are initialized with a `SynthBuilder` object which defines all the actions performed on images and surfaces which are fed to the Synthesizer. The synthesizer applies are sequence of transformations to each input (e.g., an image or surface) and returns the result. The inputs, however, are only available (known) at runtime. To work around this, we provide a `Pipeline` class which, when called by a synthesizer, is provided with a dictionary of input images, surfaces, state etc. referred to as `mapped_inputs`. To select variables from this dictionary, we provide a series of `Select*` classes (e.g., `SelectImage`, `SelectState`). Consequently, these *must* be used as part of a `Pipeline` and will typically be the first entry in the list of transformations provided by a particular pipeline (e.g., to select a T1-weighted image).

Sometimes we need to initialize transformations with variables that are only available at runtime. To achieve this, we provde a `PipelineModule` class which simply defers the initialization of a transformation until runtime.

The hierarchy is something like this

    Synthesizer
        PipelineBuilder
            Pipeline
                SubPipeline
                    Transforms
                PipelineModule
                    SelectImage, SelectState, etc.
                    Transforms
                Transforms

A `Subpipeline` allows you to initialize parts of a pipeline only *once* although it is used as part of several pipelines.

## YAML Custom Tags
- `!Path` constructs a `pathlib.Path` object from the value.
- `!LabelingScheme` reads the labeling scheme from `brainsynth.config.utilities.labeling_scheme` associated with the given value.
- `!include` includes the contents of the yaml file being pointed to in the current file.

## TODOs (known issues)
- If input contains a batch dimension, it will be removed and re-added. However, even if no batch dimension is present, it will still be added in the end. I should fix that...
- Check LR flip. Are segmentation labels correctly flipped/handled?
- Add possibility of flipping along any axis?
- `photo_mode` and `exvivo` are both untested.