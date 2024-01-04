from pathlib import Path

import monai
import numpy as np
import torch

from brainsynth.transforms import Reindex
from brainsynth.spatial_utils import get_roi_center_size
from brainsynth.constants import constants, filenames

filename_subjects = lambda dataset: f"subjects_{dataset}.txt"


def get_dataloader_concatenated_and_split(
    base_dir,
    datasets,
    optional_images,
    dataset_kwargs,
    dataset_splits,
    dataloader_kwargs,
    split_rng_seed=None,
):
    """Construct a dataloader by concatenating `datasets` (e.g., ds0, ds1) and
    splitting according to `dataset_splits` (e.g., train, validation).

    Parameters
    ----------
    base_dir :
        Base directory containing datasets. A dataset is found in
        base_dir / dataset_name.
    datasets :
        List of dataset names.
    optional_images :
        Key is the name of the dataset and value is a list/tuple of the
        additional contrasts that this dataset contains and which is to be
        loaded.
    dataset_kwargs :
        Kwargs passed to dataset constructor.
    dataset_splits: dict
        Dictionary where keys are split names and values are split fractions.
        Values should sum to one.
    dataloader_kwargs:
        Kwargs passed to dataloader constructor.
    split_rng_seed:
        Seed for dataset splitting.
    """
    base_dir = Path(base_dir)

    # Individual datasets
    datasets = [
        CroppedDataset(
            base_dir / ds,
            load_dataset_subjects(base_dir, ds),
            optional_images=optional_images[ds],
            dataset_id=ds,
            return_dataset_id=True,
            **dataset_kwargs,
        )
        for ds in datasets
    ]
    # Concatenated
    dataset = torch.utils.data.ConcatDataset(datasets)
    # Split in train, validation, etc.
    dataset = split_dataset(dataset, dataset_splits, split_rng_seed)
    dataloader = {
        k: make_dataloader(v, **dataloader_kwargs) for k, v in dataset.items()
    }

    # original datasets are in
    # dataloader["split_id"].dataset.dataset.datasets
    #                        subset  concat  list of original ds
    return dataloader


def split_dataset(
    dataset: monai.data.Dataset, splits: dict, rng_seed: None | int = None
):
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
    dataset: monai.data.Dataset,
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
            sampler=torch.utils.data.DistributedSampler(dataset),
        )
    else:
        kwargs |= dict(
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    return monai.data.DataLoader(dataset, **kwargs)


def write_dataset_subjects(data_dir, dataset, exclude=None):
    """Write a file called `{dataset}.txt` in `data_dir`. Dataset is the name
    of a subdirectory of `data_dir` containing the data for this dataset. E.g.,

        /my/data_dir/
            dataset0/
                sub-01
                sub-02
            dataset1/
                sub-01
                sub-02

    and

        write_dataset_subjects(data_dir, dataset0)
        write_dataset_subjects(data_dir, dataset1)

    will create the following text files

        /my/data_dir/
            subjects_dataset0.txt
            subjects_dataset1.txt

    Parameters
    ----------
    exclude :
        Names of subjects to exclude.

    """
    data_dir = Path(data_dir)
    p = data_dir / dataset
    exclude = exclude or []
    subjects = [i.name for i in sorted(p.glob("*")) if i not in exclude]

    # subjects = list(p.glob("*"))
    # for subject in exclude:
    #     subjects.remove(subject)

    np.savetxt(data_dir / filename_subjects(dataset), subjects, fmt="%s")


def load_dataset_subjects(data_dir, dataset):
    return np.genfromtxt(
        Path(data_dir) / filename_subjects(dataset), dtype="str"
    ).tolist()


def get_spatial_crop(fov_center, fov_size, fov_pad, hemi_bbox=None, shape=None):
    # ignore center and pad
    if fov_center == "image" and fov_size == "image":
        return None

    center, size = get_roi_center_size(fov_center, fov_size, fov_pad, hemi_bbox, shape)

    return monai.transforms.SpatialCrop(center, size)


class CroppedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        subjects: None | list | tuple,
        default_images: None | list | tuple = None,
        optional_images: None | list | tuple = None,
        onehot_encoding: None | dict = None,
        surface_resolution: None | int = 6,
        surface_hemi: None | list | str | tuple = "both",
        fov_center: str | list[int] | tuple[int] = "image",
        fov_size: str | list[int] | tuple[int] = "image",
        fov_pad: int | list[int] | tuple[int] = 0,
        dataset_id: None | str = None,
        return_dataset_id: bool = False,
        rng_seed: None | int = None,
    ):
        """This Dataset does not include synthesis of any kind but simply loads
        the required MR images from disk.

        Will read the specified surface resolution (and hemisphere depending on
        bounding box argument)

        subject_dir/

            info.pt
            surface.{resolution}.{hemi}.pt
            t1.nii
            segmentation.nii
            ...




        Parameters
        ----------
        subjects:
            If None, then glob `dataset_dir`. If a string, then try to load
            using numpy.loadtxt. This should be a file containing a list of
            subjects to use. If tuple/list, it is a list of subject names.
        fov_center: "brain", "image", "lh", "rh", list/tuple of coordinates
            Default = "center".
        fov_size: "brain", "image", "lh", "rh", list/tuple of length along each dimension
            Default = "image".
        fov_pad: int | list[int] | tuple[int]
            Default = 0. Padding is applied to both sides, e.g., fov_pad = 2
            results in fov_size increasing by 4.
        """
        self.dataset_id = dataset_id or ""
        self.return_dataset_id = return_dataset_id
        self.dataset_dir = Path(dataset_dir)
        match subjects:
            case None:
                subjects = list(self.dataset_dir.glob("*"))
            case str():
                subjects = np.loadtxt(subjects).tolist()
            # else expect a list/tuple of subject names
        self.subjects = subjects

        self.default_images = (
            list(filenames.default_images._fields)
            if default_images is None
            else list(default_images)
        )
        self.optional_images = [] if optional_images is None else list(optional_images)

        self.surface_resolution = surface_resolution

        match surface_hemi:
            case None:
                surface_hemi = []
            case list() | tuple():
                assert all(
                    h in constants.HEMISPHERES for h in surface_hemi
                ), "Invalid arguments to `surface_hemi`"
            case "both":
                surface_hemi = constants.HEMISPHERES
            case "lh" | "rh":
                surface_hemi = [surface_hemi]
        self.surface_hemi = surface_hemi

        if onehot_encoding is None:
            onehot_encoding = {}

        self.fov_center = (
            fov_center if isinstance(fov_center, str) else torch.tensor(fov_center)
        )
        self.fov_size = (
            fov_size if isinstance(fov_size, str) else torch.tensor(fov_size)
        )
        if isinstance(fov_pad, int):
            fov_pad = torch.IntTensor([fov_pad]).expand(3)
        self.fov_pad = 2 * fov_pad

        # Transformations
        self.load_image = monai.transforms.LoadImage(
            reader="NibabelReader", ensure_channel_first=True
        )
        self.as_onehot = {
            img: monai.transforms.Compose(
                [
                    monai.transforms.EnsureType(dtype=torch.int),
                    Reindex(torch.IntTensor(labels)),
                    monai.transforms.AsDiscrete(to_onehot=len(labels)),
                ]
            )
            for img, labels in onehot_encoding.items()
        }

        if rng_seed is not None:
            torch.random.manual_seed(rng_seed)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        if self.return_dataset_id:
            return self.dataset_id, *self.load_data(
                self.dataset_dir / self.subjects[idx]
            )
        else:
            return self.load_data(self.dataset_dir / self.subjects[idx])

    def get_spatial_crop(self, info):
        return get_spatial_crop(
            self.fov_center, self.fov_size, self.fov_pad, info["bbox"], info["shape"]
        )

    def load_surfaces(self, subject_dir):
        if self.surface_resolution is None:
            return {}, {}
        else:
            if self.surface_hemi == "random":
                surface_hemi = [constants.HEMISPHERES[torch.randint(0, 2, (1,))]]
            else:
                surface_hemi = self.surface_hemi
            surfaces = {
                h: {
                    k: monai.data.MetaTensor(v)
                    for k, v in torch.load(
                        subject_dir / filenames.surfaces[self.surface_resolution, h]
                    ).items()
                }
                for h in surface_hemi
            }
            # load the initial resolution
            r = constants.SURFACE_RESOLUTIONS[0]
            initial_vertices = {
                h: monai.data.MetaTensor(
                    torch.load(subject_dir / filenames.surface_templates[r, h])
                )
                for h in surface_hemi
            }

            return surfaces, initial_vertices

    def load_images(self, subject_dir, spatial_crop=None):
        images = {}

        for image in self.default_images:
            filename = subject_dir / getattr(filenames.default_images, image)
            images[image] = self._load_single_image(image, filename, spatial_crop)

        for image in self.optional_images:
            filename = subject_dir / getattr(filenames.optional_images, image)
            images[image] = self._load_single_image(image, filename, spatial_crop)

        return images

    def _load_single_image(self, image_name, image_filename, spatial_crop=None):
        # if f.exists():
        img = self.load_image(image_filename)
        if spatial_crop is not None:
            img = spatial_crop(img)
        if image_name in self.as_onehot:
            img = self.as_onehot[image_name](img)
        return img

    def load_data(self, subject_dir):
        info = torch.load(subject_dir / filenames.info)

        spatial_crop = self.get_spatial_crop(info)

        images = self.load_images(subject_dir, spatial_crop)
        surfaces, initial_vertices = self.load_surfaces(subject_dir)

        return images, surfaces, initial_vertices, info
