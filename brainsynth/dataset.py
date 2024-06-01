from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from brainsynth.constants import IMAGE, SURFACE

filename_subjects = lambda dataset: f"{dataset}.txt"
filename_subjects_subset = lambda dataset, subset: f"{dataset}_{subset}.txt"


def atleast_4d(tensor):
    return atleast_4d(tensor[None]) if tensor.ndim < 4 else tensor


def load_dataset_subjects(filename):
    return np.genfromtxt(filename, dtype="str").tolist()


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds_dir: str | Path,
        ds_name: str,
        subjects: None | str | list | tuple = None,
        synthesizer=None,
        images: None | list | tuple = None,
        ds_structure: str = "flat",
        target_surface_resolution: None | int = 6,
        target_surface_hemispheres: None | list | str | tuple = "both",
        # initial_surface_type: str = "template",  # or prediction
        initial_surface_resolution: int = 0,
    ):
        """

        Will read the specified surface resolution (and hemisphere depending on
        bounding box argument)

        subject_dir/

            info.pt
            surface.{resolution}.{hemi}.pt
            t1.nii
            segmentation.nii
            ...



        from brainsynth.dataset import *
        surface_hemi = "both"
        dataset_id = "HCP"
        surface_resolution = 4
        surface_hemi = "both"
        alternative_synth = ("norm", )
        optional_images = [] #("T1",)
        default_images = ("generation", "segmentation", "norm")
        subjects = "train"
        data_dir = Path("/mnt/scratch/personal/jesperdn/datasets/")
        dataset = "HCP"
        synthesizer=None

        ds = SynthesizedDataset(
            ds_dir="/mnt/projects/CORTECH/nobackup/training_data",
            ds_name = "HCP",
            subjects=["sub-001", "sub-010"],
            synthesizer=None,
            images=["generation_labels", "brainseg", "t1w"],
            ds_structure="flat",
            target_surface_resolution=5,
            target_surface_hemispheres="lh",
            initial_surface_resolution=0,
        )
        a,b,c=ds[0]



        Parameters
        ----------
        subjects:
            If None, then glob `dataset_dir`. If a string, then try to load
            using numpy.loadtxt. This should be a file containing a list of
            subjects to use. If tuple/list, it is a list of subject names.
        synthesizer:
            Configured synthesizer
        """
        self._initialize_io_settings(ds_dir, ds_name, ds_structure, subjects)
        self._initialize_image_settings(images)
        self._initialize_target_surface_settings(
            target_surface_resolution, target_surface_hemispheres
        )
        # self._initialize_initial_surface_settings(
        #     initial_surface_resolution, initial_surface_type
        # )
        self.initial_surface_resolution = initial_surface_resolution

        if synthesizer:
            assert synthesizer.device == torch.device("cpu")
        self.synthesizer = synthesizer

    def _initialize_io_settings(self, ds_dir, ds_name, ds_structure, subjects):
        self.ds_dir = Path(ds_dir)
        self.ds_name = ds_name
        self.ds_structure = ds_structure

        # Subjects
        self.subjects = (
            load_dataset_subjects(subjects) if isinstance(subjects, str) else subjects
        )

        # Filename generators
        match self.ds_structure:
            case "flat":

                def get_image_filename(subject, image):
                    return self.ds_dir / f"{self.ds_name}.{subject}.{image}"

                def get_surface_filename(subject, surface):
                    # return self.ds_dir / f"{self.ds_name}.{subject}.{surface}"
                    return self.ds_dir / f"{self.ds_name}.{subject}.surf_dir" / surface

            case "tree":

                def get_image_filename(subject, image):
                    return self.ds_dir / self.ds_name / subject / image

                def get_surface_filename(subject, surface):
                    return self.ds_dir / self.ds_name / subject / surface

        self.get_image_filename = get_image_filename
        self.get_surface_filename = get_surface_filename

    def _initialize_image_settings(self, images):
        self.images = list(IMAGE.images._fields) if images is None else list(images)

    def _initialize_target_surface_settings(
        self, target_surface_resolution, target_surface_hemispheres
    ):
        self.target_surface_resolution = target_surface_resolution

        match target_surface_hemispheres:
            case None:
                target_surface_hemispheres = []
            case list() | tuple():
                assert all(
                    h in SURFACE.hemispheres for h in target_surface_hemispheres
                ), "Invalid arguments to `surface_hemi`"
            case "both":
                target_surface_hemispheres = SURFACE.hemispheres
            case "lh" | "rh":
                target_surface_hemispheres = [target_surface_hemispheres]
        self.target_surface_hemispheres = target_surface_hemispheres

    # def _initialize_initial_surface_settings(
    #     self, initial_surface_resolution, initial_surface_type
    # ):
    #     self.initial_surface_resolution = initial_surface_resolution

    #     match initial_surface_type:
    #         case "template":

    #             def initial_surface_loader(subject_dir, hemispheres, surfaces=None):
    #                 res = constants.SURFACE_RESOLUTIONS[self.initial_surface_resolution]
    #                 return {
    #                     hemi: monai.data.MetaTensor(
    #                         torch.load(
    #                             subject_dir / constants.surface_template[hemi, res]
    #                         )
    #                     )
    #                     for hemi in hemispheres
    #                 }

    #         case "prediction":

    #             def initial_surface_loader(subject_dir, hemispheres, surfaces):
    #                 res = constants.SURFACE_RESOLUTIONS[self.initial_surface_resolution]
    #                 return {
    #                     hemi: {
    #                         surf: monai.data.MetaTensor(
    #                             torch.load(
    #                                 subject_dir
    #                                 / constants.surface_prediction[hemi, surf, res]
    #                             )
    #                         )
    #                         for surf in surfaces
    #                     }
    #                     for hemi in hemispheres
    #                 }

    #         case _:
    #             raise ValueError

    #     self.load_initial_surfaces = initial_surface_loader

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return self.load_data(self.subjects[idx])

    def load_data(self, subject_dir):
        images = self.load_images(subject_dir)
        surfaces, initial_vertices = self.load_surfaces(subject_dir)

        if self.synthesizer is None:
            return images, surfaces, initial_vertices
        else:
            with torch.no_grad():
                return self.synthesizer(images, surfaces, initial_vertices)

    def load_images(self, subject):
        images = {}
        for image in self.images:
            img = getattr(IMAGE.images, image)

            # Not all subjects have all images
            if (fi := self.get_image_filename(subject, img.filename)).exists():
                images[image] = self.load_image(fi, img.dtype)
                # If the image has an associated defacing mask, load it
                if img.defacingmask:
                    mask = getattr(IMAGE.images, img.defacingmask)
                    if (fm := self.get_image_filename(subject, mask.filename)).exists():
                        images[img.defacingmask] = self.load_image(fm, mask.dtype)
        return images

    def load_image(self, filename, dtype):
        # Images seem to be (x,y,z,c) or (x,y,z) but we want (c,x,y,z)
        return atleast_4d(
            torch.from_numpy(nib.load(filename).dataobj[:]).to(dtype=dtype).squeeze()
        ).contiguous()

    def load_surfaces(self, subject):
        if self.target_surface_resolution is None:
            return {}, {}
        else:
            # Select hemisphere
            if self.target_surface_hemispheres == "random":
                surface_hemi = [SURFACE.hemispheres[torch.randint(0, 2, (1,))]]
            else:
                surface_hemi = self.target_surface_hemispheres

            target_surfaces = self._load_target_surfaces(subject, surface_hemi)
            initial_vertices = self._load_initial_surfaces(subject, surface_hemi)

            return target_surfaces, initial_vertices

    def _load_target_surfaces(self, subject, hemi):
        r = self.target_surface_resolution
        return {
            h: {
                t: torch.load(
                    self.get_surface_filename(
                        subject, SURFACE.files.target[h, t, r]
                    )
                )
                for t in SURFACE.types
            }
            for h in hemi
        }

    def _load_initial_surfaces(self, subject, hemi):
        r = SURFACE.resolutions[self.initial_surface_resolution]
        return {
            h: torch.load(
                self.get_surface_filename(subject, SURFACE.files.template[h, r])
            )
            for h in hemi
        }

    # def load_surfaces(self, subject_dir):
    #     if self.target_surface_resolution is None:
    #         return {}, {}
    #     else:
    #         if self.target_surface_hemispheres == "random":
    #             surface_hemi = [constants.HEMISPHERES[torch.randint(0, 2, (1,))]]
    #         else:
    #             surface_hemi = self.target_surface_hemispheres

    #         # load target surfaces
    #         surfaces = self.load_target_surfaces(subject_dir, surface_hemi)

    #         # load initial surfaces
    #         surfs = tuple(
    #             surfaces[surface_hemi[0]].keys()
    #         )  # assumes surfaces are the same in each hemisphere!
    #         initial_vertices = self.load_initial_surfaces(
    #             subject_dir, surface_hemi, surfs
    #         )

    #         return surfaces, initial_vertices

    # def load_info(self, subject_dir):
    #     return torch.load(subject_dir / constants.info)


def make_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 1,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    shuffle: bool = True,
    pin_memory: bool = True,
    distributed: bool = False,
):
    kwargs = dict(batch_size=batch_size, pin_memory=pin_memory)
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
    return torch.utils.data.DataLoader(dataset, **kwargs)


def setup_dataloader(
    subjects: dict,
    synthesizer,
    dataset_kwargs,
    dataloader_kwargs,
):
    """Construct a dataloader by concatenating `datasets` (e.g., ds0, ds1).

    Parameters
    ----------
    base_dir :
        Base directory containing datasets. A dataset is found in
        base_dir / dataset_name.
    datasets :
        List of dataset names.
    dataset_kwargs :
        Kwargs passed to dataset constructor.
    dataloader_kwargs:
        Kwargs passed to dataloader constructor.
    """
    dataset_kwargs = dataset_kwargs or {}
    dataloader_kwargs = dataloader_kwargs or {}

    # Individual datasets
    concat_ds = torch.utils.data.ConcatDataset(
        [
            AugmentedDataset(
                subjects=load_dataset_subjects(subjects[k]),
                synthesizer=synthesizer,
                **v,
            )
            for k, v in dataset_kwargs.items()
        ]
    )
    # original datasets are in
    # dataloader.dataset.dataset.datasets
    #                        subset  concat  list of original ds
    return make_dataloader(concat_ds, **dataloader_kwargs)


# def get_dataloader_concatenated_and_split(
#     base_dir,
#     datasets,
#     optional_images,
#     dataset_kwargs,
#     dataset_splits,
#     dataloader_kwargs,
#     split_rng_seed=None,
# ):
#     """Construct a dataloader by concatenating `datasets` (e.g., ds0, ds1) and
#     splitting according to `dataset_splits` (e.g., train, validation).

#     Parameters
#     ----------
#     base_dir :
#         Base directory containing datasets. A dataset is found in
#         base_dir / dataset_name.
#     datasets :
#         List of dataset names.
#     optional_images :
#         Key is the name of the dataset and value is a list/tuple of the
#         additional contrasts that this dataset contains and which is to be
#         loaded.
#     dataset_kwargs :
#         Kwargs passed to dataset constructor.
#     dataset_splits: dict
#         Dictionary where keys are split names and values are split fractions.
#         Values should sum to one.
#     dataloader_kwargs:
#         Kwargs passed to dataloader constructor.
#     split_rng_seed:
#         Seed for dataset splitting.
#     """
#     base_dir = Path(base_dir)

#     # Individual datasets
#     datasets = [
#         CroppedDataset(
#             base_dir / ds,
#             # load_dataset_subjects(base_dir, ds),
#             optional_images=optional_images[ds],
#             dataset_id=ds,
#             return_dataset_id=True,
#             **dataset_kwargs,
#         )
#         for ds in datasets
#     ]
#     # Concatenated
#     dataset = torch.utils.data.ConcatDataset(datasets)
#     # Split in train, validation, etc.
#     dataset = split_dataset(dataset, dataset_splits, split_rng_seed)
#     dataloader = {
#         k: make_dataloader(v, **dataloader_kwargs) for k, v in dataset.items()
#     }

#     # original datasets are in
#     # dataloader["split_id"].dataset.dataset.datasets
#     #                        subset  concat  list of original ds
#     return dataloader


def split_dataset(
    dataset: torch.utils.data.Dataset, splits: dict, rng_seed: None | int = None
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


def write_dataset_subjects(
    data_dir,
    out_dir,
    subsets: None | dict = None,
    exclude: None | dict = None,
    rng_seed: int | None = None,
):
    """Write a file called `{dataset}.txt` for each dataset in `data_dir`.
    `data_dir` is structured as follows

        /data_dir/
            dataset0.sub-01.image1.nii
            dataset0.sub-01.image2.nii
            ...
            dataset0.sub-02.image1.nii
            dataset0.sub-02.image2.nii
            ...
            dataset1.sub-01.image1.nii
            dataset1.sub-01.image2.nii
            ...
            etc.


    # datset` is the name of a subdirectory of `data_dir` containing the data for this dataset. E.g.,



        /my/data_dir/
            dataset0/
                sub-01
                sub-02
            dataset1/
                sub-01
                sub-02


        data_dir = "/mnt/projects/CORTECH/nobackup/training_data"
        out_dir = "/mnt/projects/CORTECH/nobackup/training_data_subjects"


    Parameters
    ----------
    data_dir : _type_
        _description_
    out_dir : _type_
        _description_
    subsets : None | dict, optional
        _description_, by default None
    exclude : None | dict, optional
        _description_, by default None
    rng_seed : int | None, optional
        _description_, by default None
    """
    data_dir = Path(data_dir)
    exclude = exclude or {}
    out_dir = Path(out_dir) if out_dir is not None else data_dir
    if not out_dir.exists():
        out_dir.mkdir()

    subjects = {}
    prev_sub = None
    for i in sorted(data_dir.glob("*")):
        ds, sub, *_ = i.name.split(".")
        if prev_sub == sub:
            continue
        if ds not in exclude or sub not in exclude[ds]:
            try:
                subjects[ds].append(sub)
            except KeyError:
                subjects[ds] = [sub]
        prev_sub = sub

    for ds, subs in subjects.items():
        np.savetxt(out_dir / f"{ds}.txt", subs, fmt="%s")

    if subsets:
        for ds, subs in subjects.items():
            arr = make_subsets(subs, subsets, rng_seed)
            for name, subarr in zip(subsets, arr):
                np.savetxt(out_dir / f"{ds}_{name}.txt", subarr, fmt="%s")


def make_subsets(subs, subsets: dict, rng_seed: int | None = None):
    fractions = np.cumsum(list(subsets.values()))
    assert np.isclose(fractions[-1], 1.0)
    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    rng = np.random.default_rng(rng_seed)
    rng.shuffle(subs)
    n = len(subs)
    index = [np.round(n * f).astype(int) for f in fractions[:-1]]
    return np.split(subs, index)


# def write_dataset_subjects(
#     data_dir,
#     dataset,
#     subsets: dict | None = None,
#     exclude: list | tuple | None = None,
#     rng_seed: int | None = None,
# ):
#     """Write a file called `{dataset}.txt` in `data_dir`. Dataset is the name
#     of a subdirectory of `data_dir` containing the data for this dataset. E.g.,

#         /my/data_dir/
#             dataset0/
#                 sub-01
#                 sub-02
#             dataset1/
#                 sub-01
#                 sub-02

#     and

#         write_dataset_subjects(data_dir, dataset0)
#         write_dataset_subjects(data_dir, dataset1)

#     will create the following text files

#         /my/data_dir/
#             subjects_dataset0.txt
#             subjects_dataset1.txt

#     Parameters
#     ----------
#     exclude :
#         Names of subjects to exclude.

#     """
#     data_dir = Path(data_dir)
#     p = data_dir / dataset
#     exclude = exclude or []
#     subjects = [i.name for i in sorted(p.glob("*")) if i not in exclude]

#     np.savetxt(data_dir / filename_subjects(dataset), subjects, fmt="%s")

#     if subsets:
#         for ds, subs in subjects.items():
#             names, arr =  make_subsets(subs, subsets, rng_seed)

#             for name, subarr in zip(names, arr):
#                 np.savetxt(
#                     out_dir / filename_subjects_subset(dataset, name), subarr, fmt="%s"
#                 )
