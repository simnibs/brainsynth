from pathlib import Path

from brainsynth.constants import IMAGE


def subjects_subset_str(p: Path, ds: str, subset: None | str = None):
    if subset is None:
        dd = p / f"{ds}.txt"
    else:
        dd = p / f"{ds}.{subset}.txt"
    return dd if dd.exists() else None


# ds_config = DefaultDatasetConfig(
#    "/mnt/projects/CORTECH/nobackup/training_data",
#    "/mnt/projects/CORTECH/nobackup/training_data_subjects",
#    "train",
# )
# ds_config.get_dataset_kwargs()


class XDatasetConfig:
    def __init__(
        self,
        root_dir: Path | str,
        subject_dir: Path | str,
        subject_subset: None | str,
        datasets: None | list | tuple = None,
        ds_structure="flat",
    ) -> None:
        known_datasets = (
            "ABIDE",
            "ADHD200",
            "ADNI3",
            "AIBL",
            "Buckner40",
            "Chinese-HCP",
            "COBRE",
            "HCP",
            "ISBI2015",
            "MCIC",
            "OASIS3",
        )

        datasets = datasets or known_datasets

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
                ds_structure=ds_structure,
            )
            for ds in datasets
        }


class DatasetConfig:
    def __init__(
        self,
        root_dir: Path | str,
        subject_dir: Path | str,
        subject_subset: None | str = None,
        datasets: None | list | tuple = None,
        images: None | list | tuple = None,
        exclude_subjects: None | str = None,
        raise_on_invalid_image: bool = True,
        **kwargs,
    ):
        known_datasets = (
            "ABIDE",
            "ADHD200",
            "ADNI-GO2",
            "ADNI3",
            "AIBL",
            "Buckner40",
            "Chinese-HCP",
            "COBRE",
            "HCP",
            "ISBI2015",
            "MCIC",
            "OASIS3",
        )

        valid_images = {
            "ABIDE": IMAGE.default_images,
            "ADHD200": IMAGE.default_images,
            "ADNI-GO2": ["t1w", "flair"],
            "ADNI3": IMAGE.default_images + ["flair"],
            "AIBL": IMAGE.default_images + ["flair"],
            "Buckner40": IMAGE.default_images,
            "Chinese-HCP": IMAGE.default_images,
            "COBRE": IMAGE.default_images,
            "HCP": IMAGE.default_images + ["t2w"],
            "ISBI2015": IMAGE.default_images,
            "MCIC": IMAGE.default_images,
            "OASIS3": IMAGE.default_images + ["ct", "t2w"],
        }
        datasets = datasets or known_datasets
        images = ("generation_labels",) if images is None else images
        use_images = {}
        for ds in datasets:

            def _filter_func(i):
                val = i in valid_images[ds]
                if not val and raise_on_invalid_image:
                    raise ValueError(
                        f"Invalid image `{i}` for dataset {ds}. Valid images are {valid_images[ds]}"
                    )
                return val

            use_images[ds] = list(filter(_filter_func, images))

        root_dir = Path(root_dir)
        subject_dir = Path(subject_dir)

        ds_is_known = [ds in known_datasets for ds in datasets]
        assert all(
            ds_is_known
        ), f"Unknown dataset(s) {[ds for ds, i in zip(datasets, ds_is_known) if not i]} (from {known_datasets})"

        self.dataset_kwargs = {}
        for ds in datasets:
            sub_str = subjects_subset_str(subject_dir, ds, subject_subset)
            if sub_str is not None:
                self.dataset_kwargs[ds] = dict(
                    root_dir=root_dir,
                    name=ds,
                    subjects=sub_str,
                    exclude_subjects=subjects_subset_str(
                        subject_dir, ds, exclude_subjects
                    )
                    if exclude_subjects is not None
                    else None,
                    images=use_images[ds],
                    **kwargs,
                )
