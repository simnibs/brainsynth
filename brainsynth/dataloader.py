import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def split_dataset(dataset: Dataset, splits: dict, rng_seed: None | int = None):
    """_summary_

    Parameters
    ----------
    dataset : Dataset
        _description_
    splits : dict
        _description_
    rng_seed : None | int, optional
        _description_, by default None

    Returns
    -------
    dict
        Dictionary of subsets of dataset (.indices contains indices of the data
        in a given subset).
    """
    names = list(splits.keys())
    fractions = list(splits.values())
    if rng_seed is not None:
        torch.manual_seed(rng_seed)
    return dict(zip(names, torch.utils.data.random_split(dataset, fractions)))

# def split_dataset(
#         subjects: list | tuple | npt.NDArray,
#         split: dict,
#         split_fraction: list | tuple | npt.NDArray,
#         names: list | tuple,
#         seed: None | int = None
#     ):
#     """Split `subjects` into several arrays according to `split_fraction`. The
#     elements of `subjects` are permuted before the splitting is performed.

#     Parameters
#     ----------
#     subjects : _type_
#         _description_
#     split: dict
#         The fractions at which to split `subjects`, e.g.,

#             {train: 0.8, validation: 0.2}
#             {train: 0.7, validation: 0.2, test: 0.1}

#     split_fraction : list | tuple | ...
#         The fractions at which to split `subjects`, e.g., [0.8] or [0.7, 0.9]
#         will result in two or three subarrays, respectively.
#     names : list | tuple
#         Names of the resulting subarrays, e.g., ["train", "validation"] or
#         ["train", "validation", "test"].
#     seed : _type_, optional
#         _description_, by default None

#     Returns
#     -------
#     _type_
#         _description_
#     """
#     split_fractions = np.array(list(split.values()))
#     assert split_fractions.sum() == 1.0
#     assert len(names) == len(split_fraction) + 1
#     split_fraction = np.array(split_fraction)
#     assert np.all(split_fraction > 0) and np.all(split_fraction < 1)
#     n_subjects = len(subjects)

#     rng = np.random.default_rng(seed)
#     perm = rng.permutation(n_subjects)

#     splitter = np.round(np.array(split_fraction) * n_subjects).astype(int)

#     return dict(zip(names, np.array_split(subjects[perm], splitter)))


def make_dataloader(
    dataset: Dataset,
    batch_size=1,
    shuffle=True,
    num_workers=2,
    prefetch_factor=2,
    distributed=False,
):
    kwargs = dict(batch_size=batch_size)
    if distributed:
        kwargs |= dict(
            shuffle=False,
            sampler=DistributedSampler(dataset),
        )
    else:
        kwargs |= dict(
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    return DataLoader(dataset, **kwargs)
