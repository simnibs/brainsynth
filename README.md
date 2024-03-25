# BrainSynth
Brain scan synthesizer.

## YAML Custom Tags
- `!Path` constructs a `pathlib.Path` object from the value.
- `!LabelingScheme` reads the labeling scheme from `brainsynth.config.utilities.labeling_scheme` associated with the given value.
- `!include` includes the contents of the yaml file being pointed to in the current file.

## TODOs (known issues)
- If input contains a batch dimension, it will be removed and re-added. However, even if no batch dimension is present, it will still be added in the end. I should fix that...
- Check LR flip. Are segmentation labels correctly flipped/handled?
- Add possibility of flipping along any axis?
- `photo_mode` and `exvivo` are both untested.