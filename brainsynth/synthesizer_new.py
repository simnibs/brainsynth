import torch

from brainsynth.constants.constants import mapped_input_keys as mikeys
from brainsynth.config.augmentation_configuration import AugmentationConfiguration
import brainsynth.config.pipelines
from brainsynth.transforms import EnsureDevice, InputSelectorError, Pipeline


class Synthesizer(torch.nn.Module):
    def __init__(self, config: AugmentationConfiguration):
        super().__init__()
        self.config = config
        self.ensure_device = EnsureDevice(self.config.device)

        self.pipeline = getattr(brainsynth.config.pipelines, config.pipeline)()

    def execute(self, pipelines: dict, mapped_inputs: dict, out: dict | None = None):
        out = out if out is not None else {}
        for k, pipeline in pipelines.items():
            print(k)
            try:
                if isinstance(pipeline, Pipeline):
                    out[k] = pipeline(mapped_inputs)
                else:
                    out[k] = pipeline()
            except InputSelectorError:
                # could not find some input
                pass
        return out

    def forward(
        self,
        images: dict[str, torch.Tensor],
        surfaces: dict[str, dict[str, torch.Tensor]],
        initial_vertices: dict[str, torch.Tensor]
    ):

        mapped_inputs = {
            mikeys.image: self.ensure_device(images),
            mikeys.initial_vertices: self.ensure_device(initial_vertices),
            mikeys.surface: self.ensure_device(surfaces),
            mikeys.state: {},
        }

        # Build the pipelines
        state_pipeline, output_pipeline = self.pipeline.build(self.config)

        # Set the state
        mapped_inputs["state"] = self.execute(
            state_pipeline, mapped_inputs, mapped_inputs[mikeys.state]
        )

        # Generate the output
        return self.execute(output_pipeline, mapped_inputs)


config = AugmentationConfiguration(device="cpu", alternative_images=("T1w", "T2w"), segmentation_labels=(1,2,3,4))
self = Synthesizer(config)

images=dict(
    T1w=torch.rand((256, 256, 256)),
    T2w=torch.rand((256, 256, 256)),
    generation=torch.randint(0, 20, (256,256,256)),
    segmentation=torch.randint(0, 4, (256,256,256)),
    T1w_mask=torch.randint(0, 1, (256,256,256), dtype=torch.int),
    T2w_mask=torch.randint(0, 1, (256,256,256), dtype=torch.int),
    # FLAIR=torch.rand((10,10,10))
)
surfaces=dict(lh=dict(white=torch.rand((100,3)), pial=torch.rand((100,3))))
initial_vertices=dict(lh=torch.rand((60,3)))

self(images, surfaces, initial_vertices)
