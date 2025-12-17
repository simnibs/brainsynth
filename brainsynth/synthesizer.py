import torch

from brainsynth.constants import mapped_input_keys as mik
from brainsynth.config import builders
from brainsynth.config import SynthesizerConfig
from brainsynth.config.utilities import SynthesizerOutput
from brainsynth.transforms import EnsureDevice
from brainsynth.transforms.utils import recursive_function
from brainsynth.utilities import squeeze_nd

recursive_squeeze_nd = recursive_function(squeeze_nd)


class Synthesizer(torch.nn.Module):
    def __init__(self, config: SynthesizerConfig):
        super().__init__()
        self.config = config
        self.ensure_device = EnsureDevice(self.config.device)

        self.builder = getattr(builders, config.builder)(config)

    def forward(
        self,
        images: dict[str, torch.Tensor],
        affines: dict[str, torch.Tensor] | None = None,
        surfaces: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> SynthesizerOutput:
        surfaces = surfaces or {}
        affines = affines or {}

        # Remove batch dim

        images = recursive_squeeze_nd(images, n=4, dim=0)
        if affines is not None:
            affines = recursive_squeeze_nd(affines, n=2, dim=0)
        if surfaces is not None:
            surfaces = recursive_squeeze_nd(surfaces, n=2, dim=0)

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
