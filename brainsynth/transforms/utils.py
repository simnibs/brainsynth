import functools


# def channel_last(x, as_contiguous: bool = False):
#     out = x.permute((1, 2, 3, 0))
#     out = out.contiguous() if as_contiguous else out
#     return out


def channel_last(x):
    n = len(x.size())
    match n:
        case 4:
            # assume unbatched
            return x.permute((1, 2, 3, 0))
        case 5:
            # assume dim 0 is batch dim
            return x.permute((0, 2, 3, 4, 1))
        case _:
            raise NotImplementedError("Only implemented for 4-D and 5-D tensors.")


def recurse_in_dict(func):
    @functools.wraps(func)
    def wrapper(x, *args, **kwargs):
        if isinstance(x, dict):
            return {k: wrapper(v, *args, **kwargs) for k, v in x.items()}
        else:
            return func(x, *args, **kwargs)

    return wrapper


def method_recursion_dict(func):
    @functools.wraps(func)
    def wrapper(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return {k: wrapper(self, v, *args, **kwargs) for k, v in x.items()}
        else:
            return func(self, x, *args, **kwargs)

    return wrapper
