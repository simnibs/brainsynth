import torch

from brainsynth.constants import mapped_input_keys as mikeys
from brainsynth.config import SynthesizerConfig
import brainsynth.config.synthesizer_builder
from brainsynth.transforms import EnsureDevice, Pipeline


class Synthesizer(torch.nn.Module):
    def __init__(self, config: SynthesizerConfig):
        super().__init__()
        self.config = config
        self.ensure_device = EnsureDevice(self.config.device)

        self.builder = getattr(brainsynth.config.synthesizer_builder, config.builder)(config)

    def execute(self, pipelines: dict, mapped_inputs: dict, out: dict | None = None):
        out = out if out is not None else {}
        for k, pipeline in pipelines.items():
            res = pipeline(mapped_inputs) if isinstance(pipeline, Pipeline) else pipeline()
            # If a pipeline was skipped (e.g., because the input did not exist,
            # then `res` is None)
            if res is not None:
                out[k] = res
        return out

    @staticmethod
    def unpack(out):
        surface = out.pop("surface") if "surface" in out else {}
        initial_vertices = out.pop("initial_vertices") if "initial_vertices" in out else {}
        affine = out.pop("affine") if "affine" in out else {}
        return out, surface, initial_vertices, affine

    def forward(
        self,
        images: dict[str, torch.Tensor],
        surfaces: dict[str, dict[str, torch.Tensor]] | None = None,
        initial_vertices: dict[str, torch.Tensor] | None = None,
        affines: dict[str, torch.Tensor] | None = None,
        unpack: bool = True,
    ):
        surfaces = surfaces or {}
        initial_vertices = initial_vertices or {}
        affines = affines or {}
        mapped_inputs = {
            mikeys.image: self.ensure_device(images),
            mikeys.initial_vertices: self.ensure_device(initial_vertices),
            mikeys.surface: self.ensure_device(surfaces),
            mikeys.affine: self.ensure_device(affines),
            mikeys.state: {},
            "output": {},
        }

        # Build the pipelines
        state_pipeline, output_pipeline = self.builder.build()

        # Set the internal state
        _ = self.execute(
            state_pipeline, mapped_inputs, mapped_inputs[mikeys.state]
        )

        # Generate the output
        out = self.execute(output_pipeline, mapped_inputs, mapped_inputs["output"])

        return self.unpack(out) if unpack else out
