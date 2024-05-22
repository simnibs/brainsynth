from typing import Any

import torch


class Pipeline(torch.nn.Module):
    def __init__(self, *transforms):
        super().__init__()
        self.transforms = list(transforms)

    def initialize(self, input_selector):
        for i,t in enumerate(self.transforms):
            if isinstance(t, Pipeline):
                t.initialize(input_selector)
            elif isinstance(t, Module):
                    self.transforms[i] = t(input_selector)
            # else assume regular transform

    def forward(self):
        x = self.transforms[0]()
        for transform in self.transforms[1:]:
            x = transform(x)
        return x


class Module:
    def __init__(self, transform: BaseTransform, *args, **kwargs) -> None:
        self.transform = transform
        self.args = args or []
        self.kwargs = kwargs or {}

    def check_input(
            self,
            input_selector: InputSelector,
            inp,
            inp_suffix = "InputSelector:"
        ) -> Any:
        # if isinstance(inp, SequentialTransform):
        #     for t in inp.transforms:
        #         t = self.check_input(input_selector, t)
        if isinstance(inp, Module):
            inp = inp(input_selector)()
        elif isinstance(inp, str) and inp.startswith(inp_suffix):
            inp = input_selector(inp.removeprefix(inp_suffix))()
        return inp

    def check_transform(
            self,
            input_selector: InputSelector,
        ):
        if isinstance(self.transform, InputSelector):
            self.transform = input_selector
        else:
            assert issubclass(self.transform, BaseTransform)

    def __call__(self, inp_sel: InputSelector):
        self.check_transform(inp_sel)
        args = [self.check_input(inp_sel, arg) for arg in self.args]
        kwargs = {k: self.check_input(inp_sel, v) for k,v in self.kwargs.items()}
        return self.transform(*args, **kwargs)


class BaseTransform(torch.nn.Module):
    def __init__(self, device: None | torch.device = None) -> None:
        super().__init__()
        self.device = torch.device("cpu") if device is None else device


class ReturnTransform(BaseTransform):
    def __init__(self, x):
        super().__init__()
        self.x = x
    def forward(self):
        return self.x

class InputSelector(BaseTransform):
    def __init__(
        self,
        images: dict[str, torch.Tensor],
        surfaces: dict[str, dict[str, torch.Tensor]],
        initial_vertices: dict[str, torch.Tensor],
        state: dict[str, torch.Tensor],
    ):
        super().__init__()
        self.mapped_inputs = dict(
            image=images,
            surface=surfaces,
            initial_vertices=initial_vertices,
            state=state,
        )

    def recursive_selection(self, mapped_input, keys):
        selection = mapped_input[keys[0]]
        if len(keys) == 1:
            return selection
        else:
            return self.recursive_selection(selection, keys[1:])

    def forward(self, selection: str):
        """

        image:T1w
        image:segmentation  grabs ["segmentation"] from the dict of input images
        surface:lh:white    grabs ["lh"]["white"] from the dict of input surfaces

        state
            in_size         input size
            out_size        output field-of-view
            grid            image grid (same shape as out_size)
            scale           the average scaling of the inputs (from the linear transformation)

        """
        osel = selection.split(":")
        return ReturnTransform(
            self.recursive_selection(self.mapped_inputs[osel[0]], osel[1:])
        )



# class SequentialTransform(BaseTransform):
#     def __init__(self, *transforms):
#         super().__init__()
#         self.transforms = tuple(transforms)

#     def forward(self, x = None):
#         for transform in self.transforms:
#             x = transform() if x is None else transform(x)
#         return x


class RandomizableTransform(BaseTransform):
    """Similar to monai.transforms.RandomizableTransform but using torch."""

    def __init__(self, prob: float = 1.0, device: None | torch.device = None):
        super().__init__(device)
        assert 0 <= prob <= 1.0, "Invalid probability"
        self.prob = prob
        self.randomize()

    def randomize(self) -> None:
        """ """
        self._do_transform = torch.rand(1, device=self.device) < self.prob

    def should_apply_transform(self):
        return self._do_transform


class RandomChoice(BaseTransform):
    def __init__(
        self,
        args: tuple | list,
        prob: tuple | list | None = None,
        device: None | torch.device = None,
    ) -> None:
        super().__init__(device)
        n = len(args)
        if prob is None:
            self.prob = torch.full((n,), 1/n, device = device)
        else:
            self.prob = torch.tensor(prob, device=device)
        assert self.prob.sum() == 1.0
        assert len(args) == len(self.prob)
        self.choices = args

    def forward(self):
        return self.choices[torch.multinomial(self.prob, 1)]
