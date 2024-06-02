from pathlib import Path

import torch

from brainsynth.constants import IMAGE
from brainsynth.transforms.spatial import GridCentering


class SynthesizerConfig:
    def __init__(
        self,
        pipeline: str = "DefaultPipeline",
        in_res: list[float] | tuple[float, float, float] = (1.0, 1.0, 1.0),
        out_size: list[int] | tuple[int, int, int] = (192, 192, 192),
        align_corners: bool = True,
        out_center_str: str = "image",
        segmentation_labels: list[int] | tuple | str = "brainseg",
        photo_mode: bool = False,
        photo_spacing_range: list[float] | tuple[float, float] = (2.0, 7.0),
        photo_thickness: float = 0.001,
        alternative_images: None | list[str] | tuple = None,
        device: str | torch.device = "cuda",
    ):
        self.device = torch.device(device)

        self.pipeline = pipeline

        assert len(in_res) == 3
        self.in_res = torch.tensor(in_res, device=self.device)

        assert len(out_size) == 3
        self.out_size = torch.tensor(out_size, device=self.device)
        assert torch.all(self.out_size % 2 == 0), "Output FOV should be divisible by 2."

        self.align_corners = align_corners
        assert (
            align_corners == True
        ), "Spatial augmentation is untested with align_corners = False and probably does not work properly."
        self.grid = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(s.item(), dtype=torch.float, device=self.device)
                    for s in self.out_size
                ],
                indexing="ij",
            ),
            dim=-1,
        )
        # center the grid
        grid_centering = GridCentering(self.out_size, self.align_corners, self.device)
        self.center = grid_centering.center
        self.centered_grid = grid_centering(self.grid)

        assert out_center_str in {"brain", "image", "lh", "random", "rh"}
        self.out_center_str = out_center_str

        if isinstance(segmentation_labels, str):
            segmentation_labels = getattr(IMAGE.labeling_scheme, segmentation_labels)
        self.segmentation_labels = segmentation_labels
        self.segmentation_num_labels = len(self.segmentation_labels)

        self.photo_mode = photo_mode

        assert len(photo_spacing_range) == 2
        self.photo_spacing_range = photo_spacing_range
        self.photo_thickness = photo_thickness

        # if alternative_images is not None:
        # alternative_images = tuple(f"{mikeys.image}:{i}" for i in alternative_images)
        self.alternative_images = alternative_images

    def __repr__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items()])


def subjects_subset_str(p, ds, subset=None):
    if subset is None:
        return p / f"{ds}.txt"
    else:
        return p / f"{ds}.{subset}.txt"


# ds_config = DefaultDatasetConfig(
#    "/mnt/projects/CORTECH/nobackup/training_data",
#    "/mnt/projects/CORTECH/nobackup/training_data_subjects",
#    "train",
# )
# ds_config.get_dataset_kwargs()


class DatasetConfig:
    def __init__(
        self,
        data_dir: Path | str,
        subject_dir: Path | str,
        subject_subset: None | str,
        synthesizer=None,
        datasets: None | list | tuple = None,
        images: dict | None = None,
        ds_structure="flat",
        target_surface_resolution=5,
        target_surface_hemispheres="both",
        initial_surface_resolution=0,
    ):
        known_datasets = (
            "ABIDE",
            "ADHD200",
            "ADNI3",
            "AIBL",
            "bif",
            "Buckner40",
            "Chinese-HCP",
            "COBRE",
            "HCP",
            "ISBI2015",
            "MCIC",
            "OASIS3",
        )
        default_images = {
            "ABIDE": ["t1w"],
            "ADHD200": ["t1w"],
            "ADNI3": ["flair", "t1w"],
            "AIBL": ["flair", "t1w"],
            "bif": ["t1w"],
            "Buckner40": ["t1w"],
            "Chinese-HCP": ["t1w"],
            "COBRE": ["t1w"],
            "HCP": ["t1w", "t2w"],
            "ISBI2015": ["t1w"],
            "MCIC": ["t1w"],
            "OASIS3": ["ct", "t1w", "t2w"],
        }

        datasets = datasets or known_datasets
        images = images or {}

        data_dir = Path(data_dir)
        subject_dir = Path(subject_dir)

        ds_is_known = [ds in known_datasets for ds in datasets]
        assert all(
            ds_is_known
        ), f"Unknown dataset(s) {[ds for ds, i in zip(datasets, ds_is_known) if not i]} (from {known_datasets})"

        self.dataset_kwargs = {
            ds: dict(
                data_dir=data_dir,
                ds_name=ds,
                subjects=subjects_subset_str(subject_dir, ds, subject_subset),
                synthesizer=synthesizer,
                images=images[ds] if ds in images else default_images[ds],
                ds_structure=ds_structure,
                target_surface_resolution=target_surface_resolution,
                target_surface_hemispheres=target_surface_hemispheres,
                initial_surface_resolution=initial_surface_resolution,
            )
            for ds in datasets
        }
