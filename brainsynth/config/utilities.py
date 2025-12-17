# from pathlib import Path
# from types import SimpleNamespace

# import yaml

from dataclasses import dataclass
import torch

from brainsynth.transforms import Pipeline

# from brainsynth import root_dir
from brainsynth.constants import mapped_input_keys as mikeys
from brainsynth.config.synthesizer import SynthesizerConfig
from brainsynth.transforms.utils import recursive_function
# from brainsynth.constants import IMAGE


# def load_config(filename=None):
#     config_dir = root_dir / "config"
#     filename = filename or config_dir / "synthesizer.yaml"
#     filename = Path(filename)
#     if not filename.exists():
#         filename = config_dir / filename

#     with open(filename, "r") as f:
#         config = yaml.load(f, Loader=get_loader())
#     return recursive_dict_to_namespace(config)


# def get_loader():
#     loader = yaml.SafeLoader
#     loader.add_constructor("!Path", path_constructor)
#     loader.add_constructor("!LabelingScheme", labeling_scheme_constructor)
#     loader.add_constructor("!include", include_constructor)
#     return loader


# def path_constructor(loader, node):
#     """ """
#     return Path(loader.construct_scalar(node))


# def labeling_scheme_constructor(loader, node):
#     return getattr(IMAGE.labeling_scheme, loader.construct_scalar(node))


# def include_constructor(loader, node):
#     # os.path.join(os.path.dirname(loader.name), node.value)
#     with open(Path(loader.name).parent / loader.construct_scalar(node)) as f:
#         return yaml.load(f, Loader=yaml.SafeLoader)


# def recursive_dict_to_namespace(d):
#     namespace = d
#     if isinstance(d, dict):
#         namespace = SimpleNamespace(**d)
#         for k, v in namespace.__dict__.items():
#             setattr(namespace, k, recursive_dict_to_namespace(v))
#     return namespace


# def recursive_namespace_to_dict(ns):
#     if isinstance(ns, SimpleNamespace):
#         return {k: recursive_namespace_to_dict(v) for k, v in vars(ns).items()}
#     else:
#         return ns


class Pipelines(dict):
    def __init__(self, *args, **kwargs):
        """Collection of pipelines in a dict style layout that can be executed
        in sequence.
        """
        super().__init__(*args, **kwargs)

    def execute(self, mapped_inputs: dict, key: str | tuple, prune_none: bool = True):
        """Execute the pipelines. The output is stored in `mapped_inputs` which
        means that previously computed items are available for subsequent
        steps.
        """
        for k, pipeline in self.items():
            match pipeline:
                case Pipeline():
                    res = pipeline(mapped_inputs)
                case Pipelines():
                    if prune_none:
                        res = {}
                        for kk, vv in pipeline.items():
                            out = vv(mapped_inputs)
                            if out is not None:
                                res[kk] = out
                    else:
                        res = {kk: vv(mapped_inputs) for kk, vv in pipeline.items()}
                case None:
                    res = None
                case _:
                    res = pipeline()
            # If a pipeline was skipped (e.g., because the input did not exist,
            # then `res` is None)
            mapped_inputs[key][k] = res
        return mapped_inputs[key]


class StatePipelines(Pipelines):
    """Pipelines which write to the `state` entry of `mapped_inputs`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self, mapped_inputs: dict):
        # If a pipeline was skipped (e.g., because the input did not exist,
        # then `res` is None)
        return super().execute(mapped_inputs, mikeys.state)


@dataclass
class OutputPipelines(Pipelines):
    """Pipelines which write to the `output` entry of `mapped_inputs`."""

    image: Pipeline | None = None
    affine: Pipeline | None = None
    images: Pipelines | None = None
    surfaces: Pipeline | Pipelines | None = None

    def __post_init__(self):
        super().__init__(**vars(self))

    def execute(self, mapped_inputs: dict):
        out = super().execute(mapped_inputs, mikeys.output)
        return SynthesizerOutput(**out)


@dataclass
class SynthesizerOutput(dict):
    image: torch.Tensor
    affine: torch.Tensor | None = None
    images: dict[str, torch.Tensor] | None = None
    surfaces: dict[str, dict[str, torch.Tensor]] | None = None

    def unsqueeze(self, dim: int = 0):
        r_unsqueeze = recursive_function(torch.unsqueeze)

        self.image = self.image.unsqueeze(dim)
        if self.affine is not None:
            self.affine = self.affine.unsqueeze(dim)
        if self.images is not None:
            self.images = r_unsqueeze(self.images, dim)
        if self.surfaces is not None:
            self.surfaces = r_unsqueeze(self.surfaces, dim)


class SynthBuilder:
    def __init__(self, config: SynthesizerConfig) -> None:
        self.config = config
        self.device = config.device

    def initialize_spatial_transforms(self):
        """Optional."""
        pass

    def build_spatial_transforms(self, *args, **kwargs):
        raise NotImplementedError

    def build_intensity_transforms(self, *args, **kwargs):
        raise NotImplementedError

    def build_resolution_transforms(self, *args, **kwargs):
        raise NotImplementedError

    def build_state(self) -> StatePipelines:
        raise NotImplementedError

    def build_output(self) -> OutputPipelines:
        raise NotImplementedError

    def build(self):
        """Build a full synthesizer pipeline by collecting multiple Pipeline
        instances.

        Parameters
        ----------
        config : SynthesizerConfig


        Returns
        -------
        state :
            A dictionary of Pipelines/transformations that updates the state of
        the synthesizer each time it is called.
        output :
            A dictionary of Pipelines/transformations that generates the
            outputs, e.g., a synthesized image and processed T1, T2, surfaces,
            etc.
        """

        self.initialize_spatial_transforms()
        state = self.build_state()

        # The following relies (may rely) on the state
        self.build_spatial_transforms(**self.config.spatial_transforms_kw)
        self.build_intensity_transforms(**self.config.intensity_transforms_kw)
        self.build_resolution_transforms(**self.config.resolution_transforms_kw)

        output = self.build_output()

        return state, output
