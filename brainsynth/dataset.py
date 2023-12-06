from pathlib import Path

import monai
import numpy as np
import torch

from brainsynth.transforms import Reindex
from brainsynth.spatial_utils import get_roi_center_size
from brainsynth.constants import filenames
from brainsynth.constants.constants import HEMISPHERES

filename_subjects = lambda dataset: f"subjects_{dataset}.txt"

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
    return np.genfromtxt(Path(data_dir) / filename_subjects(dataset), dtype="str").tolist()


def get_spatial_crop(fov_center, fov_size, fov_pad, hemi_bbox=None, shape=None):

    # ignore center and pad
    if fov_center == "image" and fov_size == "image":
        return None

    center, size = get_roi_center_size(
        fov_center, fov_size, fov_pad, hemi_bbox, shape
    )

    return monai.transforms.SpatialCrop(center, size)


class CroppedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        subjects: None | list | tuple,
        default_images: None | list | tuple = None,
        optional_images: None | list | tuple = None,
        surface_resolution: None | int = 6,
        surface_hemi: None | list | str | tuple = "both",
        fov_center: str | list[int] | tuple[int] = "image",
        fov_size: str | list[int] | tuple[int] = "image",
        fov_pad: int | list[int] | tuple[int] = 0,
        images_as_one_hot: None | dict = None,
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
        self.dataset_dir = Path(dataset_dir)
        match subjects:
            case None:
                subjects = list(self.dataset_dir.glob("*"))
            case str():
                subjects = np.loadtxt(subjects).tolist()
            # else expect a list/tuple of subject names
        self.subjects = subjects

        self.default_images = list(filenames.default_images._fields) if default_images is None else list(default_images)
        self.optional_images = [] if optional_images is None else list(optional_images)

        self.surface_resolution = surface_resolution

        match surface_hemi:
            case None:
                surface_hemi = []
            case list() | tuple():
                assert all(h in HEMISPHERES for h in surface_hemi), "Invalid arguments to `surface_hemi`"
            case "both":
                surface_hemi = HEMISPHERES
            case "lh" | "rh":
                surface_hemi = [surface_hemi]
        self.surface_hemi = surface_hemi

        if images_as_one_hot is None:
            images_as_one_hot = {}

        self.fov_center = fov_center if isinstance(fov_center, str) else torch.tensor(fov_center)
        self.fov_size = fov_size if isinstance(fov_size, str) else torch.tensor(fov_size)
        if isinstance(fov_pad, int):
            fov_pad = torch.IntTensor([fov_pad]).expand(3)
        self.fov_pad = 2*fov_pad

        # Transformations
        self.load_image = monai.transforms.LoadImage(
            reader="NibabelReader", ensure_channel_first=True
        )
        self.as_onehot = {
            img: monai.transforms.Compose(
                [
                    monai.transforms.EnsureType(dtype=torch.int),
                    Reindex(torch.IntTensor(labels)),
                    monai.transforms.AsDiscrete(to_onehot=len(labels))
                ]
            ) for img,labels in images_as_one_hot.items()
        }

        if rng_seed is not None:
            torch.random.manual_seed(rng_seed)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        print(self.dataset_dir / self.subjects[idx])
        return self.load_data(self.dataset_dir / self.subjects[idx])

    def get_spatial_crop(self, info):
        return get_spatial_crop(
            self.fov_center, self.fov_size, self.fov_pad, info["bbox"], info["shape"]
        )

    def load_surfaces(self, subject_dir):
        if self.surface_resolution is None:
            return {}
        else:
            if self.surface_hemi == "random":
                surface_hemi = [HEMISPHERES[torch.randint(0, 2, (1, ))]]
            else:
                surface_hemi = self.surface_hemi
            return {h: torch.load(
                subject_dir / filenames.surfaces[self.surface_resolution, h]
            ) for h in surface_hemi}

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
        surfaces = self.load_surfaces(subject_dir)

        return images, surfaces, info
