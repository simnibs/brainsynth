import torch

from brainsynth.constants import mapped_input_keys as mikeys
from brainsynth.config.augmentation import AugmentationConfig
import brainsynth.config.pipelines
from brainsynth.transforms import EnsureDevice, Pipeline


class Synthesizer(torch.nn.Module):
    def __init__(self, config: AugmentationConfig):
        super().__init__()
        self.config = config
        self.ensure_device = EnsureDevice(self.config.device)

        self.pipeline = getattr(brainsynth.config.pipelines, config.pipeline)()

    def execute(self, pipelines: dict, mapped_inputs: dict, out: dict | None = None):
        out = out if out is not None else {}
        for k, pipeline in pipelines.items():
            res = pipeline(mapped_inputs) if isinstance(pipeline, Pipeline) else pipeline()
            # If
            if res is not None:
                out[k] = res
        return out

    @staticmethod
    def unpack(out):
        surface = out.pop("surface") if "surface" in out else {}
        initial_vertices = out.pop("initial_vertices") if "initial_vertices" in out else {}
        return out, surface, initial_vertices

    def forward(
        self,
        images: dict[str, torch.Tensor],
        surfaces: dict[str, dict[str, torch.Tensor]],
        initial_vertices: dict[str, torch.Tensor],
        unpack = True,
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
        out = self.execute(output_pipeline, mapped_inputs)

        return self.unpack(out) if unpack else out