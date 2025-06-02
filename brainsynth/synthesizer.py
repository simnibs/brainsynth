import torch

from brainsynth.constants import mapped_input_keys as mik
from brainsynth.config import SynthesizerConfig
import brainsynth.config.synthesizer_builder
from brainsynth.transforms import EnsureDevice, Pipeline

from brainsynth.config.synthesizer_builder import SynthesizerOutput


class Synthesizer(torch.nn.Module):
    def __init__(self, config: SynthesizerConfig):
        super().__init__()
        self.config = config
        self.ensure_device = EnsureDevice(self.config.device)

        self.builder = getattr(brainsynth.config.synthesizer_builder, config.builder)(
            config
        )

    # def execute(self, pipelines: dict, mapped_inputs: dict, out: dict | None = None):
    #     out = out if out is not None else {}
    #     for k, pipeline in pipelines.items():
    #         res = (
    #             pipeline(mapped_inputs)
    #             if isinstance(pipeline, Pipeline)
    #             else pipeline()
    #         )
    #         # If a pipeline was skipped (e.g., because the input did not exist,
    #         # then `res` is None)
    #         if res is not None:
    #             out[k] = res
    #     return out

    # @staticmethod
    # def unpack(d: dict):
    #     out = tuple()
    #     if "affine" in d:
    #         out += (d.pop("affine"),)
    #     if "initial_vertices" in d:
    #         out += (d.pop("initial_vertices"),)
    #     if "surface" in d:
    #         out += (d.pop("surface"),)
    #     return (d,) + out
    #     # surface = d.pop("surface") if "surface" in d else {}
    #     # initial_vertices = d.pop("initial_vertices") if "initial_vertices" in d else {}
    #     # affine = d.pop("affine") if "affine" in d else {}
    #     # return d, surface, initial_vertices, affine

    def forward(
        self,
        images: dict[str, torch.Tensor],
        affines: dict[str, torch.Tensor] | None = None,
        surfaces: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> SynthesizerOutput:
        surfaces = surfaces or {}
        affines = affines or {}
        mapped_inputs = {
            mik.image: self.ensure_device(images),
            mik.affine: self.ensure_device(affines),
            mik.surface: self.ensure_device(surfaces),
            mik.state: {},
            mik.output: {},
        }
        # Build the pipelines
        state, output = self.builder.build()

        # Set the internal state
        _ = state.execute(mapped_inputs)

        # Generate the output
        return output.execute(mapped_inputs)
