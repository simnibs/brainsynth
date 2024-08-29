from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
import torch

from brainsynth.constants import IMAGE, SURFACE

from brainsynth.config import DatasetConfig, SynthesizerConfig, XDatasetConfig
from brainsynth.synthesizer import Synthesizer

def atleast_4d(tensor):
    return atleast_4d(tensor[None]) if tensor.ndim < 4 else tensor


def load_dataset_subjects(filename):
    return np.genfromtxt(filename, dtype="str").tolist()

def _load_image(filename, dtype, transform: Callable | None = None):
    # Images seem to be (x,y,z,c) or (x,y,z) but we want (c,x,y,z)
    data = torch.from_numpy(nib.load(filename).dataobj[:])
    data = data if transform is None else transform(data)
    data = data.to(dtype=dtype)

    if data.ndim < 3:
        raise ValueError(f"Image {filename} has less than three dimensions (shape is {data.shape})")
    elif data.ndim == 3: # (x,y,z) -> (c,x,y,z)
        data = atleast_4d(data)
    elif data.ndim == 4: # (x,y,z,c) -> (c,x,y,z)
        data = data.permute((3,0,1,2)).contiguous()
    elif data.ndim > 4:
        raise ValueError(f"Image {filename} has more than four dimensions (shape is {data.shape})")
    return data


class SynthesizedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        name: str,
        subjects: None | str | list | tuple = None,
        synthesizer: None | Synthesizer | SynthesizerConfig = None,
        images: None | list | tuple = None,
        load_mask: bool = False,
        ds_structure: str = "flat",
        target_surface_resolution: None | int = 6,
        target_surface_hemispheres: None | list | str | tuple = "both",
        # initial_surface_type: str = "template",  # or prediction
        initial_surface_resolution: None | int = 0,
        xdataset: None | XDatasetConfig = None, # or XDataset
    ):
        """

        ds = SynthesizedDataset(
            ds_dir="/mnt/projects/CORTECH/nobackup/training_data",
            name = "HCP",
            subjects=["sub-001", "sub-010"],
            synthesizer=None,
            images=["generation_labels", "brainseg", "t1w"],
            ds_structure="flat",
            target_surface_resolution=5,
            target_surface_hemispheres="lh",
            initial_surface_resolution=0,
        )
        a,b,c=ds[0]


        xsub_kwargs : dict | None
            Dictionary with keys `root_dir`, `name`, and `subjects`.

            Optionally, key `fields` is used to specify which deformation
            fields to load. The default is

                dict(
                    self=("forward", "backward"),
                    other=("forward", "backward")
                )

            which loads forward and backward deformation fields for both
            subject to be warped from (self) and to (other).

        Parameters
        ----------
        subjects:
            If None, then glob `dataset_dir`. If a string, then try to load
            using numpy.loadtxt. This should be a file containing a list of
            subjects to use. If tuple/list, it is a list of subject names.
        synthesizer:
            Configured synthesizer
        load_mask : bool
            For each image, if it has an associated mask (e.g., defacing mask),
            load this as well.
        """
        self.load_mask = load_mask
        self._initialize_io_settings(root_dir, name, ds_structure, subjects)
        self._initialize_image_settings(images)
        self._initialize_target_surface_settings(
            target_surface_resolution, target_surface_hemispheres
        )
        # self._initialize_initial_surface_settings(
        #     initial_surface_resolution, initial_surface_type
        # )
        self.initial_surface_resolution = initial_surface_resolution

        match synthesizer:
            case Synthesizer():
                self.synthesizer = synthesizer
            case SynthesizerConfig():
                self.synthesizer = Synthesizer(synthesizer)
            case None:
                self.synthesizer = None

        if self.synthesizer is not None:
            assert self.synthesizer.device == torch.device("cpu")

        match xdataset:
            case None:
                self.xdataset = None
            case XDataset():
                self.xdataset = xdataset
            case XDatasetConfig():
                self.xdataset = torch.utils.data.ConcatDataset(
                    [XDataset(**kw) for kw in xdataset.dataset_kwargs.values()]
                )
            case _:
                raise ValueError("Wrong type for `xdataset`")

    def _initialize_io_settings(self, ds_dir, name, ds_structure, subjects):
        self.ds_dir = Path(ds_dir)
        self.name = name
        self.ds_structure = ds_structure

        # Subjects
        self.subjects = (
            load_dataset_subjects(subjects) if isinstance(subjects, (Path, str)) else subjects
        )
        self.n_subjects = len(self.subjects)

        # Filename generators
        match self.ds_structure:
            case "flat":
                def get_image_filename(subject, image):
                    return self.ds_dir / f"{self.name}.{subject}.{image}"
                def get_surface_filename(subject, surface):
                    # return self.ds_dir / f"{self.name}.{subject}.{surface}"
                    return self.ds_dir / f"{self.name}.{subject}.surf_dir" / surface
            case "tree":
                def get_image_filename(subject, image):
                    return self.ds_dir / self.name / subject / image
                def get_surface_filename(subject, surface):
                    return self.ds_dir / self.name / subject / surface

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

    def load_data(self, subject):
        images = self.load_images(subject)
        surfaces, initial_vertices = self.load_surfaces(subject)

        if self.xdataset is not None:
            # select a random subject from xdataset
            idx = torch.randint(0, len(self.xdataset), (1, ))
            ximages = self.xdataset[idx]
            # to distinguish from "self"
            images |= {f"other:{k}":v for k,v in ximages.items()}

        if self.synthesizer is None:
            return images, surfaces, initial_vertices
        else:
            with torch.no_grad():
                return self.synthesizer(images, surfaces, initial_vertices)

    def load_images(self, subject):
        images = {}
        for image in self.images:
            # Not all subjects have all images
            img = getattr(IMAGE.images, image)
            if (fi := self.get_image_filename(subject, img.filename)).exists():
                images[image] = _load_image(fi, img.dtype, img.transform)
                # If the image has an associated defacing mask, load it
                if self.load_mask and img.defacingmask is not None:
                    mask = getattr(IMAGE.images, img.defacingmask)
                    if (fm := self.get_image_filename(subject, mask.filename)).exists():
                        images[img.defacingmask] = _load_image(fm, mask.dtype)

        return images


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
                    self.get_surface_filename(subject, SURFACE.files.target[h, t, r])
                )
                for t in SURFACE.types
            }
            for h in hemi
        }

    def _load_initial_surfaces(self, subject, hemi):
        if self.initial_surface_resolution is None:
            return {}
        else:
            r = SURFACE.resolutions[self.initial_surface_resolution]
            return {
                h: torch.load(
                    self.get_surface_filename(subject, SURFACE.files.template[h, r])
                )
                for h in hemi
            }

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name} in {self.ds_dir/self.name}"

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


class XDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        name: str,
        subjects: None | str | list | tuple = None,
        ds_structure: str = "flat",
    ):
        """A dataset which loads forward and backward deformations for a
        particular subject.

        Inherits from SynthesizedDataset but hardcodes some settings which
        makes it appropriate for cross-subject morphing.
        """
        self._initialize_io_settings(root_dir, name, ds_structure, subjects)

        images = ["mni152_nonlin_backward", "mni152_nonlin_forward"]
        self._initialize_image_settings(images)


    def _initialize_io_settings(self, ds_dir, name, ds_structure, subjects):
        self.ds_dir = Path(ds_dir)
        self.name = name
        self.ds_structure = ds_structure

        # Subjects
        self.subjects = (
            load_dataset_subjects(subjects) if isinstance(subjects, (Path, str)) else subjects
        )
        self.n_subjects = len(self.subjects)

        # Filename generators
        match self.ds_structure:
            case "flat":
                def get_image_filename(subject, image):
                    return self.ds_dir / f"{self.name}.{subject}.{image}"
            case "tree":
                def get_image_filename(subject, image):
                    return self.ds_dir / self.name / subject / image

        self.get_image_filename = get_image_filename

    def _initialize_image_settings(self, images):
        self.images = list(IMAGE.images._fields) if images is None else list(images)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return self.load_images(self.subjects[idx])

    def load_images(self, subject):
        images = {}
        for image in self.images:
            img = getattr(IMAGE.images, image)
            fi = self.get_image_filename(subject, img.filename)
            images[image] = _load_image(fi, img.dtype, img.transform)
        return images


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

def concatenate_dataset(
    dataset_config: DatasetConfig,
    dataset_class: SynthesizedDataset = SynthesizedDataset,
):
    return torch.utils.data.ConcatDataset(
        [dataset_class(**kw) for kw in dataset_config.dataset_kwargs.values()]
    )
def setup_dataloader(
    dataset_config: DatasetConfig,
    dataloader_kwargs: dict | None = None,
    dataset_class: SynthesizedDataset  = SynthesizedDataset,
):
    """Construct a dataloader by first constructing datasets from `dataset_kwargs`
    and concatenating them.

    Parameters
    ----------
    dataset_kwargs :
        List or tuple of dicts containing kwargs with which to initialize each
        dataset.
    dataloader_kwargs:
        Kwargs passed to dataloader constructor.
    """
    # Individual datasets
    concat_ds = concatenate_dataset(dataset_config, dataset_class)

    # original datasets are in
    # dataloader.dataset.dataset.datasets
    #                        subset  concat  list of original ds
    dataloader_kwargs = dataloader_kwargs or {}
    return make_dataloader(concat_ds, **dataloader_kwargs)


def write_dataset_subjects(
    data_dir,
    out_dir,
    subsets: None | dict = None,
    exclude: None | dict = None,
    pattern: str = "*",
    suffix: None | str = None,
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
        subsets = dict(train=0.8, test=0.1, validation=0.1)

        write_dataset_subjects(data_dir, out_dir, subsets)

        pattern = "*FLAIR.nii"
        suffix = "flair"

        write_dataset_subjects(
            data_dir,
            out_dir,
            subsets,
            pattern=pattern,
            suffix=suffix,
        )


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
    for i in sorted(data_dir.glob(pattern)):
        ds, sub, *_ = i.name.split(".")
        if prev_sub == sub:
            continue
        if ds not in exclude or sub not in exclude[ds]:
            try:
                subjects[ds].append(sub)
            except KeyError:
                subjects[ds] = [sub]
        prev_sub = sub

    # for ds, subs in subjects.items():
    #     np.savetxt(out_dir / f"{ds}.txt", subs, fmt="%s")

    if subsets:
        for ds, subs in subjects.items():
            arr = make_subsets(subs, subsets, rng_seed)
            for name, subarr in zip(subsets, arr):
                parts = [ds, name]
                if suffix is not None:
                    parts += [suffix] if isinstance(suffix, str) else suffix
                # print(out_dir / (".".join(parts) + ".txt"))
                np.savetxt(out_dir / (".".join(parts) + ".txt"), subarr, fmt="%s")



def make_subsets(subs, subsets: dict, rng_seed: int | None = None):
    fractions = np.cumsum(list(subsets.values()))
    assert np.isclose(fractions[-1], 1.0)
    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    rng = np.random.default_rng(rng_seed)
    rng.shuffle(subs)
    n = len(subs)
    index = [np.round(n * f).astype(int) for f in fractions[:-1]]
    return tuple(np.sort(i) for i in np.split(subs, index))
