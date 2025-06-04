import torch

from brainsynth.constants import mapped_input_keys as mik
from brainsynth.config import SynthesizerConfig
import brainsynth.config.synthesizer_builder
from brainsynth.transforms import EnsureDevice

from brainsynth.config.synthesizer_builder import SynthesizerOutput

from brainsynth.transforms.utils import recursive_function
from brainsynth.utilities import squeeze_nd


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

        # Remove batch dim
        r_squeeze_nd = recursive_function(squeeze_nd)

        images = r_squeeze_nd(images, n=4, dim=0)
        if affines is not None:
            affines = r_squeeze_nd(affines, n=2, dim=0)
        if surfaces is not None:
            surfaces = r_squeeze_nd(surfaces, n=2, dim=0)

        mapped_inputs = {
            mik.images: self.ensure_device(images),
            mik.affines: self.ensure_device(affines),
            mik.surfaces: self.ensure_device(surfaces),
            mik.state: {},
            mik.output: {},
        }
        # Build the pipelines
        state, output = self.builder.build()

        # Set the internal state
        _ = state.execute(mapped_inputs)

        # Generate the output
        out = output.execute(mapped_inputs)
        out.unsqueeze()  # Add batch dim

        return out
