from pathlib import Path

from brainsynth.constants import IMAGE

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
        root_dir: Path | str,
        subject_dir: Path | str,
        subject_subset: None | str,
        synthesizer=None,
        datasets: None | list | tuple = None,
        images: list | tuple = None,
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

        valid_images = {
            "ABIDE":        IMAGE.default_images,
            "ADHD200":      IMAGE.default_images + ["flair"],
            "ADNI3":        IMAGE.default_images + ["flair"],
            "AIBL":         IMAGE.default_images + ["flair"],
            "bif":          IMAGE.default_images,
            "Buckner40":    IMAGE.default_images,
            "Chinese-HCP":  IMAGE.default_images,
            "COBRE":        IMAGE.default_images,
            "HCP":          IMAGE.default_images + ["t2w"],
            "ISBI2015":     IMAGE.default_images,
            "MCIC":         IMAGE.default_images,
            "OASIS3":       IMAGE.default_images + ["ct", "t2w"],
        }
        datasets = datasets or known_datasets
        images = images or ("generation_labels", )
        use_images = {}
        for ds in datasets:
            use_images[ds] = []
            for i in images:
                if i in valid_images[ds]:
                    use_images[ds].append(i)
                else:
                    raise ValueError(f"Invalid image `{i}` for dataset {ds}.")
        # use_images = {ds: [i for i in images if i in valid_images[ds]] for ds in datasets}

        root_dir = Path(root_dir)
        subject_dir = Path(subject_dir)

        ds_is_known = [ds in known_datasets for ds in datasets]
        assert all(
            ds_is_known
        ), f"Unknown dataset(s) {[ds for ds, i in zip(datasets, ds_is_known) if not i]} (from {known_datasets})"

        self.dataset_kwargs = {
            ds: dict(
                root_dir=root_dir,
                name=ds,
                subjects=subjects_subset_str(subject_dir, ds, subject_subset),
                synthesizer=synthesizer,
                images=use_images[ds],
                ds_structure=ds_structure,
                target_surface_resolution=target_surface_resolution,
                target_surface_hemispheres=target_surface_hemispheres,
                initial_surface_resolution=initial_surface_resolution,
            )
            for ds in datasets
        }
