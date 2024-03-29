from pathlib import Path

# import monai
import nibabel as nib
import numpy as np
import torch

from brainsynth.constants import constants


def atleast_4d(tensor):
    return atleast_4d(tensor[None]) if tensor.ndim < 4 else tensor



def create_dataloader(
    datasets_dir = Path | str | list[str] | tuple[str],
    datasets_name = list[str] | tuple[str],
    datasets_subjects = list[str] | tuple[str],
    dataset_kwargs=None,
    dataloader_kwargs=None,
):
    """Construct a dataloader by concatenating `datasets` (e.g., ds0, ds1).

    Example

        datasets_dir = "/path/to/data/" or ("/path/to/HCP", "path/to/OASIS3")
        datasets_names = ("HCP", "OASIS3")
        datasets_subjects = ("/path/to/HCP.subjects.train.txt", "/path/to/OASIS.subjects.train.txt")

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

        batch_size = 1
        num_workers = 2
        shuffle = True
        prefetch_factor = 2
        pin_memory = 2


    """
    if isinstance(datasets_dir, (Path, str)):
        datasets_dir = tuple(datasets_dir for _ in range(len(datasets_name)))
    datasets_dir = tuple(Path(d) for d in datasets_dir)

    dataset_kwargs = dataset_kwargs or {}
    dataloader_kwargs = dataloader_kwargs or {}

    # Individual datasets
    datasets_name = [
        AugmentedDataset(
            ds_dir, ds_name, load_dataset_subjects(ds_subjects), **dataset_kwargs,
        )
        for ds_dir, ds_name, ds_subjects in zip(datasets_dir, datasets_name, datasets_subjects)
    ]
    # Concatenated datasets
    # original datasets are in dataloader["split_id"].dataset.dataset.datasets
    #                                                 subset  concat  list of original ds
    dataset = torch.utils.data.ConcatDataset(datasets_name)
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

# def create_dataloader(
#     dataset: torch.utils.data.Dataset,
#     batch_size=1,
#     shuffle=True,
#     num_workers=2,
#     prefetch_factor=2,
#     pin_memory=True,
#     distributed=False,
# ):


def write_dataset_subjects(
        data_dir,
        dataset_id,
        out_dir=None,
        subsets: dict | None = None,
        exclude: list | tuple | None = None,
        rng_seed: int | None = None,
    ):
    """

    Example of data directory

        data_dir/
            sub-01
            sub-02
            ...

    and corresponding output with subject `train` and `validation`.

        out_dir/
            {dataset}.subjects.all.txt
            {dataset}.subjects.train.txt
            {dataset}.subjects.validation.txt

    Parameters
    ----------
    exclude :
        Names of subjects to exclude.

    """
    data_dir = Path(data_dir)
    out_dir = data_dir if out_dir is None else Path(out_dir)
    exclude = exclude or []

    subjects = [i.name for i in sorted(data_dir.glob("*")) if i not in exclude]

    np.savetxt(out_dir / f"{dataset_id}.subjects.all.txt", subjects, fmt="%s")

    if subsets:
        assert sum(subsets.values()) == 1.0
        names = list(subsets.keys())
        fractions = list(subsets.values())
        if rng_seed is not None:
            torch.manual_seed(rng_seed)

        rng = np.random.default_rng(rng_seed)
        rng.shuffle(subjects)
        n = len(subjects)
        index = [np.round(n * f).astype(int) for f in fractions[:-1]]
        arr = np.split(subjects, index)

        for name, subarr in zip(names, arr):
            np.savetxt(out_dir / f"{dataset_id}.subjects.{name}.txt", subarr, fmt="%s")


def load_dataset_subjects(filename):
    return np.genfromtxt(filename, dtype="str").tolist()


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        dataset_name: str,
        subjects: str | list | tuple,
        images: list | tuple,
        alt_images: None | list | tuple = None,
        surface_resolution: None | int = 6,
        surface_hemi: None | list | str | tuple = "both",
        transform = None,
    ):
        """

        Will read the specified surface resolution (and hemisphere dependin3g on
        bounding box argument)


        datasets_dir = "/mrhome/jesperdn/INN_JESPER/nobackup/projects/brainnet/4dk/maps"
        datasets_names = ("HCP", "Buckner40")
        datasets_subjects = ("/mrhome/jesperdn/INN_JESPER/nobackup/projects/brainnet/4dk/HCP.subjects.train.txt", "/mrhome/jesperdn/INN_JESPER/nobackup/projects/brainnet/4dk/Buckner40.subjects.train.txt")
        dataset_kwargs = dict(images=["generation_labels", "brainseg", "t1w"], surface_resolution=None, alt_images=["t1w"])
        dataloader_kwargs=dict(num_workers=1, prefetch_factor=1)
        dl = create_dataloader(datasets_dir, datasets_names, datasets_subjects, dataset_kwargs=dataset_kwargs, dataloader_kwargs=dataloader_kwargs)

        Assumes that datasets are stored data_dir

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
        alt_images = ("norm", )
        optional_images = [] #("T1",)
        default_images = ("generation", "segmentation", "norm")
        subjects = "train"
        data_dir = Path("/mnt/scratch/personal/jesperdn/datasets/")
        dataset = "HCP"
        synthesizer=None

        ds = AugmentedDataset(data_dir, dataset, subjects, synthesizer, default_images, optional_images, alternative_synth, surface_resolution, surface_hemi)

        Parameters
        ----------
        data_dir:
            Directory in which to find data.
        dataset_name:
            Name of the dataset.
        subjects:
            This should be a file containing a list of
            subjects to use. If tuple/list, it is a list of subject names.
        transform:
            A function/class (with a __call__ method) which takes in
            dictionaries corresponding to the loaded images, surfaces, and
            initial vertices, and returns a dictionary of transformed images,
            surfaces, and initial vertices (e.g., a preconfigured Synthesizer).
        """
        self.data_dir = Path(dataset_dir)
        self.dataset_name = dataset_name
        self.subjects = load_dataset_subjects(subjects) if isinstance(subjects, (Path, str)) else subjects

        if transform:
            # otherwise this does not make sense
            assert transform.device == torch.device("cpu")
        self.transform = transform

        self.images = images
        self.alt_images = [] if alt_images is None else list(alt_images)

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


    def __len__(self):
        return len(self.subjects)


    def __getitem__(self, idx):
        return self.load_data(self.subjects[idx])


    def load_data(self, subject):
        # info = self.load_info(subject_dir)
        images = self.load_images(subject)
        surfaces, initial_vertices = self.load_surfaces(subject)

        y_true_img, y_true_surf, init_vertices = self.apply_transform(images, surfaces, initial_vertices)

        # Get the synthetic image (input). If it does not exist, select a
        # random contrast from the list of alternative images
        if "synth" not in y_true_img:
            sel = torch.randint(0, len(self.alt_images), (1,))
            y_true_img["synth"] = y_true_img[self.alt_images[sel]]
        image = y_true_img.pop("synth")

        # y_true[img1], y_true[img2], y_true[surfaces][lh][white], etc.
        return image, y_true_img, y_true_surf, init_vertices


    def get_image_filename(self, subject, image):
        return self.data_dir / f"{self.dataset_name}.{subject}.{image}"


    def get_surface_filename(self, subject, surface):
        return self.data_dir / f"{self.dataset_name}.{subject}.{surface}"


    def load_image(self, filename, dtype):
        # Images seem to be (x,y,z,c) but we want (c,x,y,z)
        return atleast_4d(torch.from_numpy(nib.load(filename).dataobj[:]).to(dtype=dtype).squeeze())


    def load_images(self, subject):
        images = {}
        for image in self.images:
            img = getattr(constants.images, image)

            # All subjects do not have all images
            if (fi := self.get_image_filename(subject, img.filename)).exists():
                images[image] = self.load_image(fi, img.dtype)

                # If the image has an associated defacing mask, load it
                if img.defacingmask:
                    mask = getattr(constants.images, img.defacingmask)
                    if (fm := self.get_image_filename(subject, mask.filename)).exists():
                        images[img.defacingmask] = self.load_image(fm, mask.dtype)

        return images


    def load_surfaces(self, subject):
        if self.surface_resolution is None:
            return {}, {}
        else:
            if self.surface_hemi == "random":
                surface_hemi = [constants.HEMISPHERES[torch.randint(0, 2, (1,))]]
            else:
                surface_hemi = self.surface_hemi
            surfaces = {
                h: {
                    k: torch.tensor(v)
                    for k, v in torch.load(
                        self.get_surface_filename(subject, constants.surfaces[self.surface_resolution, h])
                    ).items()
                }
                for h in surface_hemi
            }
            # load the initial resolution
            r = constants.SURFACE_RESOLUTIONS[0]
            initial_vertices = {
                h: torch.tensor(
                    torch.load(self.get_surface_filename(subject, constants.surface_templates[r, h]))
                )
                for h in surface_hemi
            }

            return surfaces, initial_vertices


    def load_info(self, subject_dir):
        return torch.load(subject_dir / constants.info)



    def apply_transform(self, images, surfaces, initial_vertices):
        if self.transform:
            with torch.no_grad():
                y_true_img, y_true_surf, initial_vertices = self.transform(
                    images, surfaces, initial_vertices
                )
        else:
            y_true_img = images
            y_true_surf = surfaces
        return y_true_img, y_true_surf, initial_vertices
