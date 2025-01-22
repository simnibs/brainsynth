from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
import torch

from brainsynth.constants import IMAGE, SURFACE
from brainsynth.config import DatasetConfig, SynthesizerConfig, XDatasetConfig
from brainsynth.synthesizer import Synthesizer
from brainsynth.utilities import apply_affine

def atleast_4d(tensor):
    return atleast_4d(tensor[None]) if tensor.ndim < 4 else tensor


def load_dataset_subjects(filename):
    return np.atleast_1d(np.genfromtxt(filename, dtype="str")).tolist()


def _load_image(
    image: nib.Nifti1Image | Path | str, dtype, transform: Callable | None = None, return_affine: bool = False
):
    # Images seem to be (x,y,z,c) or (x,y,z) but we want (c,x,y,z)
    if isinstance(image, nib.Nifti1Image):
        img = image
    else:
        img = nib.load(image)
    data = torch.tensor(img.dataobj[:])
    data = data if transform is None else transform(data)
    data = data.to(dtype=dtype)

    if data.ndim < 3:
        raise ValueError(
            f"Image {filename} has less than three dimensions (shape is {data.shape})"
        )
    elif data.ndim == 3:  # (x,y,z) -> (c,x,y,z)
        data = atleast_4d(data)
    elif data.ndim == 4:  # (x,y,z,c) -> (c,x,y,z)
        data = data.permute((3, 0, 1, 2)).contiguous()
    elif data.ndim > 4:
        raise ValueError(
            f"Image {filename} has more than four dimensions (shape is {data.shape})"
        )
    return (data, torch.tensor(img.affine, dtype=torch.float)) if return_affine else data


class SynthesizedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        name: str,
        subjects: None | str | list | tuple = None,
        synthesizer: None | Synthesizer | SynthesizerConfig = None,
        images: None | list | tuple = None,
        load_mask: bool | str = False,
        ds_structure: str = "tree",
        target_surface: dict | None = {},
        initial_surface: dict | None = {},
        randomize_hemisphere: bool = False,
        xdataset: None | XDatasetConfig = None,  # or XDataset
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
        load_mask : bool | str
            If True, for each image, if it has an associated mask (e.g.,
            defacing mask), load this as well. Note that masks are *inverted*,
            i.e., the defacing mask encodes *invalid* voxels whereas the
            returned mask encodes the *valid* voxels! If `force` and no mask
            exists for a particular subject, make a mask where everything is
            valid. (`force` is useful when batch size > 1 as torch cannot
            collate the data if some subjects have masks and others not.)
        """
        self.load_mask = load_mask
        self._initialize_io_settings(root_dir, name, ds_structure, subjects)
        self._initialize_image_settings(images)

        self.randomize_hemisphere = randomize_hemisphere
        self._initialize_surface_settings(target_surface, initial_surface)

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
            load_dataset_subjects(subjects)
            if isinstance(subjects, (Path, str))
            else subjects
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

    def _initialize_surface_settings(
        self, target_surface_kwargs, initial_surface_kwargs
    ):
        # Target surface
        if target_surface_kwargs is None:
            self.target_surface_files = {}
        else:
            kwargs = dict(
                hemispheres=SURFACE.hemispheres,
                types=SURFACE.types,
                resolution=6,
                name="target",
            )
            if isinstance(target_surface_kwargs, dict):
                if "hemispheres" in target_surface_kwargs:
                    if target_surface_kwargs["hemispheres"] == "both":
                        target_surface_kwargs["hemispheres"] = SURFACE.hemispheres
                    elif isinstance(v := target_surface_kwargs["hemispheres"], str):
                        target_surface_kwargs["hemispheres"] = [v]
                if "types" in target_surface_kwargs:
                    v = target_surface_kwargs["types"]
                    target_surface_kwargs["types"] = [v] if isinstance(v, str) else v
                kwargs |= target_surface_kwargs
            self.hemispheres = kwargs["hemispheres"]
            self.target_surface_kwargs = kwargs
            self.target_surface_files = SURFACE.get_files(**kwargs)

        # Initial surface
        if initial_surface_kwargs is None:
            self.initial_surface_files = {}
        else:
            kwargs = dict(
                hemispheres=SURFACE.hemispheres,
                types=None,
                resolution=0,
                name="template",
            )
            if isinstance(initial_surface_kwargs, dict):
                if "hemispheres" in initial_surface_kwargs:
                    if initial_surface_kwargs["hemispheres"] == "both":
                        initial_surface_kwargs["hemispheres"] = SURFACE.hemispheres
                    elif isinstance(v := initial_surface_kwargs["hemispheres"], str):
                        initial_surface_kwargs["hemispheres"] = [v]
                if "types" in initial_surface_kwargs:
                    v = initial_surface_kwargs["types"]
                    if isinstance(v, (list, tuple)):
                        assert len(v) == 1
                        initial_surface_kwargs["types"] = v[0]
                kwargs |= initial_surface_kwargs
            assert all(i == j for i, j in zip(kwargs["hemispheres"], self.hemispheres))
            self.initial_surface_kwargs = kwargs
            self.initial_surface_files = SURFACE.get_files(**kwargs)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return self.load_data(self.subjects[idx])  # , self.subjects[idx], self.name

    def load_data(self, subject):
        images = self.load_images(subject)
        surfaces, initial_vertices = self.load_surfaces(subject)

        if self.xdataset is not None:
            # select a random subject from xdataset
            idx = torch.randint(0, len(self.xdataset), (1,))
            ximages = self.xdataset[idx]
            # to distinguish from "self"
            images |= {f"other:{k}": v for k, v in ximages.items()}

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
                if self.load_mask is not False and img.defacingmask is not None:
                    mask = getattr(IMAGE.images, img.defacingmask)
                    if (fm := self.get_image_filename(subject, mask.filename)).exists():
                        # NOTE
                        # Invert the defacing mask so that *valid* voxels are
                        # true
                        images[img.defacingmask] = ~_load_image(fm, mask.dtype)
                    elif self.load_mask == "force":
                        images[img.defacingmask] = torch.ones(
                            images[image].shape, dtype=mask.dtype
                        )
        return images

    def load_surfaces(self, subject):
        if len(self.target_surface_files) == 0:
            return {}, {}
        else:
            # Select hemisphere
            if self.randomize_hemisphere:
                surface_hemi = [SURFACE.hemispheres[torch.randint(0, 2, (1,))]]
            else:
                surface_hemi = self.hemispheres

            target_surfaces = self._load_target_surfaces(subject, surface_hemi)
            initial_vertices = self._load_initial_surfaces(subject, surface_hemi)

            return target_surfaces, initial_vertices

    # @staticmethod
    # def stack_dict(d):
    #     """dict with d[(a,b,c)] as keys to d[a][b][c]."""
    #     out = {}
    #     for k,v in d.items():
    #         this_out = out
    #         for kk in k:
    #             if kk not in this_out:
    #                 if kk == k[-1]:
    #                     this_out[kk] = v
    #                 else:
    #                     this_out[kk] = {}
    #                     this_out = this_out[kk]
    #             else:
    #                 this_out = out[kk]
    #     return out

    def _load_target_surfaces(self, subject, hemi):
        return {
            h: {
                t: torch.load(
                    self.get_surface_filename(subject, self.target_surface_files[h, t])
                )
                for t in self.target_surface_kwargs["types"]
            }
            for h in hemi
        }

    def _load_initial_surfaces(self, subject, hemi):
        if (t := self.initial_surface_kwargs["types"]) is not None:
            return {
                h: torch.load(
                    self.get_surface_filename(subject, self.initial_surface_files[h, t])
                )
                for h in hemi
            }
        else:
            return {
                h: torch.load(
                    self.get_surface_filename(subject, self.initial_surface_files[h])
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
        ds_structure: str = "tree",
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
            load_dataset_subjects(subjects)
            if isinstance(subjects, (Path, str))
            else subjects
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

class GenericDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images: list | tuple[Path, str],
        mni_transform: list | tuple[Path, str],
        mni_direction: str = "mni2sub",
        mni_space: str = "mni152",
        hemi: str | None = None,
        nv_template: int = 62,
    ):
        """Dataset formed from a list of images and transformations.


        Parameters
        ----------

        """
        assert len(images) == len(mni_transform)
        assert mni_space in ("mni152", "mni305")
        assert mni_direction in {"mni2sub", "sub2mni"}
        self.hemi = (
            tuple(
                hemi,
            )
            if isinstance(hemi, str)
            else SURFACE.hemispheres
        )

        self.images = images
        self.mni_transform = mni_transform
        self.mni_space = mni_space
        self.mni_direction = mni_direction

        self.template = {}
        self.template_meta = {}



        # FSAVERAGE
        # for h in self.hemi:
        #     v, _, m = nib.freesurfer.read_geometry(
        #         brainsynth.resources_dir / f"{h}.white.smooth", read_metadata=True
        #     )
        #     self.template[h] = torch.tensor(v[:nv_template].astype(np.float32))
        #     self.template_meta[h] = m

        # TOPOFIT
        v, _, m = nib.freesurfer.read_geometry(
            brainsynth.resources_dir / "cortex-int-lh.srf", read_metadata=True
        )
        flip_x = np.array([-1, 1, 1])
        for h in self.hemi:
            if h == "lh":
                self.template[h] = torch.tensor(v[:nv_template].astype(np.float32))
            elif h == "rh":
                self.template[h] = torch.tensor((v[:nv_template]*flip_x).astype(np.float32))
            self.template_meta[h] = m



        # (8) from https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
        mni305_to_mni152 = torch.tensor(
            [
                [0.9975, -0.0073, 0.0176, -0.0429],
                [0.0146, 1.00090003, -0.0024, 1.54960001],
                [-0.013, -0.0093, 0.9971, 1.18400002],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        match self.mni_space:
            case "mni152":
                # the template vertices are in mni305 so add a transformation
                # from mni305 to mni152
                self._preprocess_mni_transform = lambda t: self._mni_to_sub(t) @ mni305_to_mni152
            case "mni305":
                self._preprocess_mni_transform = lambda t: self._mni_to_sub(t)

    def _mni_to_sub(self, t):
        return torch.linalg.inv(t) if self.mni_direction == "sub2mni" else t

    def _load_mni_transform(self, index):
        return np.loadtxt(self.mni_transform[index])

    def prepare_initial_vertices(self, trans, vox2mri):
        trans = torch.linalg.inv(vox2mri) @ trans
        return {k: apply_affine(trans, v) for k, v in self.template.items()}

    def load_initial_vertices(self, index, vox2mri):
        trans = self._load_mni_transform(index)
        trans = torch.tensor(trans, dtype=torch.float)
        trans = self._preprocess_mni_transform(trans)
        return self.prepare_initial_vertices(trans, vox2mri)

    def preprocess_image(self, img):
        return img

    def load_image(self, index):
        img = nib.load(self.images[index])
        img = self.preprocess_image(img)
        return _load_image(img, torch.float, return_affine=True)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        image, vox2mri = self.load_image(index)
        template = self.load_initial_vertices(index, vox2mri)
        return image, template, vox2mri


def make_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 1,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    shuffle: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
    distributed: bool = False,
):
    kwargs = dict(batch_size=batch_size, drop_last=drop_last, pin_memory=pin_memory)
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


def concatenate_datasets(
    dataset_config: DatasetConfig,
    dataset_class: SynthesizedDataset = SynthesizedDataset,
):
    return torch.utils.data.ConcatDataset(
        [dataset_class(**kw) for kw in dataset_config.dataset_kwargs.values()]
    )


def setup_dataloader(
    dataset_config: DatasetConfig,
    dataloader_kwargs: dict | None = None,
    separate_datasets: bool = False,
    dataset_class: SynthesizedDataset = SynthesizedDataset,
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
    dataloader_kwargs = dataloader_kwargs or {}

    # Individual datasets
    if separate_datasets:
        dl = {
            k: make_dataloader(dataset_class(**kw), **dataloader_kwargs)
            for k, kw in dataset_config.dataset_kwargs.items()
        }
        # original datasets are in
        # dataloader.dataset.dataset.datasets
        #                        subset  concat  list of original ds
    else:
        dl = make_dataloader(
            concatenate_datasets(dataset_config, dataset_class), **dataloader_kwargs
        )
    return dl


def write_dataset_subjects(
    data_dir,
    out_dir,
    subsets: None | dict = None,
    include_datasets: None | list[str] = None,
    exclude_subjects: None | dict = None,
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


        data_dir = "/mnt/projects/CORTECH/nobackup/training_data_brainreg"
        out_dir = "/mnt/projects/CORTECH/nobackup/training_data_subjects"
        subsets = dict(train=0.8, test=0.1, validation=0.1)

        suffix = "registration"
        exclude = dict(ADNI3=["sub-075"], AIBL=["sub-046", "sub-389", "sub-601"])
        pattern = "*T1w.areg-mni*"

        write_dataset_subjects(
            data_dir,
            out_dir,
            subsets,
            exclude=exclude,
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
    out_dir = Path(out_dir) if out_dir is not None else data_dir
    if not out_dir.exists():
        out_dir.mkdir()

    exclude_subjects = exclude_subjects or {}

    # subjects = {}
    # prev_sub = None
    # for i in sorted(data_dir.glob(pattern)):
    #     ds, sub, *_ = i.name.split(".")
    #     if prev_sub == sub:
    #         continue
    #     if ds not in exclude or sub not in exclude[ds]:
    #         try:
    #             subjects[ds].append(sub)
    #         except KeyError:
    #             subjects[ds] = [sub]
    #     prev_sub = sub

    subjects = {}
    for ds in sorted(data_dir.glob("*")):
        if include_datasets is None or ds in include_datasets:
            subjects[ds] = []
            for sub in sorted((data_dir / ds).glob(pattern)):
                if ds not in exclude_subjects or sub not in exclude_subjects[ds]:
                    subjects[ds].append(sub)

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
