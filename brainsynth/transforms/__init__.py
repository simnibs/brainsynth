from typing import Type

import torch

from .base import BaseTransform, RandomChoice
from .contrast import (
    IntensityNormalization,
    RandBiasfield,
    RandBlendImages,
    RandGammaTransform,
    SynthesizeIntensityImage,
)
from .spatial import (
    RandLinearTransform,
    RandNonlinearTransform,
    RandResolution,
    SpatialCrop,
)
from .label import OneHotEncoding

class InputSelector(torch.nn.Module):
    def __init__(self, selection: str):
        super().__init__()
        self.osel = selection.split(":")

    def recursive_selection(self, mapped_input, keys):
        selection = mapped_input[keys[0]]
        if len(keys) == 1:
            return selection
        else:
            return self.recursive_selection(selection, keys[1:])

    def forward(self, mapped_inputs: dict[str, dict[str, torch.Tensor]]):
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
        return self.recursive_selection(mapped_inputs[self.osel[0]], self.osel[1:])


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


class Pipeline(torch.nn.Module):
    def __init__(self, *transforms):
        super().__init__()
        self.transforms = list(transforms)
        allowed_first_transform = (InputSelector, PipelineModule, RandomChoice)
        assert isinstance(
            self.transforms[0], allowed_first_transform
        ), f"The first transform of a Pipeline must be one of {allowed_first_transform} but got {type(self.transforms[0])}."

    def initialize(self, mapped_inputs: dict[str, dict[str, torch.Tensor]]):
        """Initialize PipelineModules and Pipelines."""
        for i, t in enumerate(self.transforms):
            if isinstance(t, Pipeline):
                t.initialize(mapped_inputs)
            elif isinstance(t, PipelineModule):
                self.transforms[i] = t(mapped_inputs)
            # else assume regular transform

    def forward(self, mapped_inputs: dict[str, dict[str, torch.Tensor]]):
        self.initialize(mapped_inputs)
        # x = self.transforms[0]()
        # for transform in self.transforms[1:]:
        #     x = transform(x)

        for t in self.transforms:
            if isinstance(t, InputSelector):
                x = t(mapped_inputs)
            elif isinstance(t, RandomChoice):
                x = t()
                if isinstance(x, Pipeline):
                    x = x(mapped_inputs)
            else:
                x = t(x)
        return x
