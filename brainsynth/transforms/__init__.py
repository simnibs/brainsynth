from typing import Type

import torch

from .base import BaseTransform, EnsureDevice, EnsureDType, RandomChoice
from .contrast import (
    IntensityNormalization,
    RandBiasfield,
    RandBlendImages,
    RandGammaTransform,
    RandMaskRemove,
    SynthesizeIntensityImage,
)
from .spatial import (
    CenterFromString,
    CheckCoordsInside,
    Grid,
    GridCentering,
    GridNormalization,
    GridSample,
    RandLinearTransform,
    RandNonlinearTransform,
    RandResolution,
    SpatialCrop,
    SpatialSize,
    SurfaceBoundingBox,
    TranslationTransform,
)
from .label import MaskFromLabelImage, OneHotEncoding
from .misc import ExtractDictKeys, Intersection, ServeValue, Uniform

from brainsynth.constants import mapped_input_keys as mikeys


class InputSelectorError(Exception):
    pass


class InputSelector(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.selection = args

    def recursive_selection(self, mapped_input, keys):
        try:
            selection = mapped_input[keys[0]]
            if len(keys) == 1:
                return selection
            else:
                return self.recursive_selection(selection, keys[1:])
        except KeyError:
            raise InputSelectorError(
                f"No key `{keys[0]}` in `mapped_input` with keys {tuple(mapped_input.keys())}."
            )

    def forward(
        self,
        mapped_inputs: dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]],
    ):
        """
            images: dict[str, torch.Tensor],
            surfaces: dict[str, dict[str, torch.Tensor]],
            initial_vertices: dict[str, torch.Tensor],
            state: dict[str, torch.Tensor],

        image:T1w
        image:segmentation  grabs ["segmentation"] from the dict of input images
        surface:lh:white    grabs ["lh"]["white"] from the dict of input surfaces

        state
            in_size         input size
            out_size        output field-of-view
            grid            image grid (same shape as out_size)
            scale           the average scaling of the inputs (from the linear transformation)

        """

        # self.mapped_inputs = dict(
        #     image=images,
        #     surface=surfaces,
        #     initial_vertices=initial_vertices,
        #     state=state,
        # )
        if len(self.selection) == 0:
            return mapped_inputs
        elif len(self.selection) == 1:
            return self.recursive_selection(mapped_inputs, self.selection)
        else:
            return self.recursive_selection(
                mapped_inputs[self.selection[0]], self.selection[1:]
            )


class SelectImage(InputSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = mikeys.image

    def forward(self, mapped_inputs: dict[str, dict[str, torch.Tensor]]):
        return super().forward(mapped_inputs[self.image])


class SelectInitialVertices(InputSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = mikeys.initial_vertices

    def forward(self, mapped_inputs: dict[str, dict[str, torch.Tensor]]):
        return super().forward(mapped_inputs[self.image])


class SelectState(InputSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = mikeys.state

    def forward(self, mapped_inputs: dict[str, dict[str, torch.Tensor]]):
        return super().forward(mapped_inputs[self.image])

class SelectSurface(InputSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = mikeys.surface

    def forward(self, mapped_inputs: dict[str, dict[str, torch.Tensor]]):
        return super().forward(mapped_inputs[self.image])


class PipelineModule:
    def __init__(self, transform: Type[torch.nn.Module], *args, **kwargs) -> None:
        self.transform = transform
        self.args = args or []
        self.kwargs = kwargs or {}

    def check_input(
        self,
        mapped_inputs: dict,
        inp,
    ):
        if isinstance(inp, InputSelector):
            # When an InputSelector is used as (kw)arg, evaluate
            inp = inp(mapped_inputs)
        elif isinstance(inp, RandomChoice):
            # When a RandomChoice is used as (kw)arg, evaluate
            inp = inp()
        elif isinstance(inp, PipelineModule):
            # When a PipelineModule is used as (kw)arg, initialize
            inp = inp(mapped_inputs)()
        return inp

    def check_transform(
        self,
    ):
        assert issubclass(self.transform, torch.nn.Module)

    def __call__(self, mapped_inputs: dict[str, dict[str, torch.Tensor]]):
        """Initialize the transform."""
        self.check_transform()
        args = [self.check_input(mapped_inputs, arg) for arg in self.args]
        kwargs = {k: self.check_input(mapped_inputs, v) for k, v in self.kwargs.items()}
        return self.transform(*args, **kwargs)


class SubPipeline(torch.nn.Module):
    def __init__(self, *transforms):
        super().__init__()
        self.transforms = list(transforms)
        self.is_initialized = False

    def initialize(self, mapped_inputs: dict[str, dict[str, torch.Tensor]], force: bool = False):
        """Initialize PipelineModules and Pipelines."""
        if self.is_initialized and not force:
            return

        for i, t in enumerate(self.transforms):
            match t:
                case SubPipeline():
                    t.initialize(mapped_inputs)
                case PipelineModule():
                    self.transforms[i] = t(mapped_inputs)
                # else assume regular transform

        self.is_initialized = True

    def reinitialize(self, mapped_inputs: dict[str, dict[str, torch.Tensor]]):
        self.initialized(mapped_inputs, force=True)

    @staticmethod
    def _run_element(t, mapped_inputs, x):
        match t:
            case SubPipeline():
                x = t(mapped_inputs, x)
            case _:
                x = t(x)
        return x

    def forward(self, mapped_inputs: dict[str, dict[str, torch.Tensor]], x):
        self.initialize(mapped_inputs)
        for t in self.transforms:
            x = self._run_element(t, mapped_inputs, x)
        return x


class Pipeline(SubPipeline):
    def __init__(
            self,
            *transforms,
            unpack_inputs: bool = True,
            skip_on_InputSelectorError: bool = False,
        ):
        super().__init__(*transforms)
        self.unpack_inputs = unpack_inputs
        self.skip_on_InputSelectorError = skip_on_InputSelectorError
        valid_first_transform = (InputSelector, ServeValue, PipelineModule, RandomChoice)
        assert isinstance(
            self.transforms[0], valid_first_transform
        ), f"The first transform of a Pipeline must be one of {valid_first_transform} but got {type(self.transforms[0])}."

    @staticmethod
    def _collect_input(transform, mapped_inputs):
        match transform:
            case InputSelector():
                x = transform(mapped_inputs)
            case RandomChoice() | ServeValue():
                x = transform()
                if isinstance(x, Pipeline):
                    x = x(mapped_inputs)
            case _:
                raise RuntimeError(f"Invalid first transform {t}")
        return x

    def _transform_element(self, transforms, mapped_inputs, x):
        for t in transforms:
            x = self._run_element(t, mapped_inputs, x)
        return x

    def _transform_input(self, transforms: list | tuple, mapped_inputs, x):
        if self.unpack_inputs:
            match x:
                case dict():
                    x = {k: self._transform_input(transforms, mapped_inputs, v) for k,v in x.items()}
                # case list() | tuple():
                #     x = [self._transform_input(transforms, mapped_inputs, v) for v in x]
                case _:
                    x = self._transform_element(transforms, mapped_inputs, x)
        else:
            x = self._transform_element(transforms, mapped_inputs, x)
        return x

    def forward(self, mapped_inputs: dict[str, dict[str, torch.Tensor]]):
        self.initialize(mapped_inputs)

        try:
            x = self._collect_input(self.transforms[0], mapped_inputs)
        except InputSelectorError as e:
            if self.skip_on_InputSelectorError:
                x = None
            else:
                raise e
        else:
            x = self._transform_input(self.transforms[1:], mapped_inputs, x)

        return x
