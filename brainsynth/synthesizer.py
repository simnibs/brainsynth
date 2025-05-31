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

        self.builder = getattr(brainsynth.config.synthesizer_builder, config.builder)(
            config
        )

    def execute(self, pipelines: dict, mapped_inputs: dict, out: dict | None = None):
        out = out if out is not None else {}
        for k, pipeline in pipelines.items():
            res = (
                pipeline(mapped_inputs)
                if isinstance(pipeline, Pipeline)
                else pipeline()
            )
            # If a pipeline was skipped (e.g., because the input did not exist,
            # then `res` is None)
            if res is not None:
                out[k] = res
        return out

    @staticmethod
    def unpack(d: dict):
        out = tuple()
        if "affine" in d:
            out += (d.pop("affine"),)
        if "initial_vertices" in d:
            out += (d.pop("initial_vertices"),)
        if "surface" in d:
            out += (d.pop("surface"),)
        return (d,) + out
        # surface = d.pop("surface") if "surface" in d else {}
        # initial_vertices = d.pop("initial_vertices") if "initial_vertices" in d else {}
        # affine = d.pop("affine") if "affine" in d else {}
        # return d, surface, initial_vertices, affine

    def forward(
        self,
        images: dict[str, torch.Tensor],
        surfaces: dict[str, dict[str, torch.Tensor]] | None = None,
        initial_vertices: dict[str, torch.Tensor] | None = None,
        affines: dict[str, torch.Tensor] | None = None,
        unpack: bool = False,
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
        _ = self.execute(state_pipeline, mapped_inputs, mapped_inputs[mikeys.state])

        # Generate the output
        out = self.execute(output_pipeline, mapped_inputs, mapped_inputs["output"])

        return self.unpack(out) if unpack else out
