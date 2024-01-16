# BrainSynth
Brain scan synthesizer.


## TODOs (known issues)
- If input contains a batch dimension, it will be removed and re-added. However, even if no batch dimension is present, it will still be added in the end. I should fix that...
- Check LR flip. Are segmentation labels correctly flipped/handled?
- Add possibility of flipping along any axis?