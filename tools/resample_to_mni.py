from pathlib import Path
import sys

import nibabel as nib
import nibabel.processing
import numpy as np
# from tqdm import tqdm

if __name__ == "__main__":

    index = int(sys.argv[1])

    out_dir = Path("/mnt/scratch/personal/jesperdn/t1_affine_transform")

    MNI152 = nib.load("/home/jesperdn/repositories/brainnet/MNI152_T1_1mm_192_224_192.nii.gz")

    import brainsynth
    from brainsynth.config.dataset import DatasetConfig
    import torch

    root_dir = Path("/mnt/projects/CORTECH/nobackup")
    ds_conf = DatasetConfig(
        root_dir = root_dir / "training_data",
        subject_dir = root_dir / "training_data_subjects",
        subject_subset=None,
        images=[],
        target_surface_resolution=5,
        target_surface_hemispheres="both",
    )

    mni_vox2ras = MNI152.affine
    mni_ras2vox = np.linalg.inv(mni_vox2ras)


    concat_ds = torch.utils.data.ConcatDataset(
        [brainsynth.dataset.SynthesizedDataset(**kw) for kw in ds_conf.dataset_kwargs.values()]
    )

    for i,cs in enumerate(concat_ds.cumulative_sizes):
        if index < cs:
            ds = concat_ds.datasets[i]
            if i > 0:
                sub_idx = index - concat_ds.cumulative_sizes[i-1]
            else:
                sub_idx = index
            break

    subject = ds.subjects[sub_idx]

    print(index, ds.name, subject)

    ras2mni = np.loadtxt(ds.ds_dir / f"{ds.name}.{subject}.mni152_affine_backward.txt")

    # image
    t1w = nib.load(ds.ds_dir / f"{ds.name}.{subject}.T1w.nii")
    vox2ras = t1w.affine
    t1w_trans = nib.Nifti1Image(
        t1w.get_fdata().squeeze().astype(t1w.get_data_dtype()),
        ras2mni @ vox2ras
    )
    out = nibabel.processing.resample_from_to(t1w_trans, MNI152)
    out.to_filename(out_dir / ".".join([ds.name, subject, "T1w.areg-mni", "nii"]))

    try:
        t1w_mask = nib.load(ds.ds_dir / f"{ds.name}.{subject}.T1w.defacingmask.nii")
        t1w_mask_trans = nib.Nifti1Image(
            t1w_mask.get_fdata().squeeze().astype(t1w_mask.get_data_dtype()),
            ras2mni @ vox2ras
        )
        out = nibabel.processing.resample_from_to(t1w_mask_trans, MNI152)
        out.to_filename(out_dir / ".".join([ds.name, subject, "T1w.areg-mni.defacingmask", "nii"]))
    except FileNotFoundError:
        pass

    # surface

    affine = torch.tensor(mni_ras2vox @ ras2mni @ vox2ras).float()

    d = ".".join([ds.name, subject, "surf_dir"])
    if not (out_dir / d).exists():
        (out_dir/d).mkdir()

    for h in ["lh", "rh"]:
        for s in ["white", "pial"]:
            name = ".".join([h,s,"5","target","pt"])

            data = torch.load(ds.ds_dir / d / name)
            out = data @ affine[:3,:3].T + affine[:3,3]
            torch.save(out, out_dir / d / name)

    # for ds in concat_ds.datasets:
    #     print(ds.name)
    #     for subject in tqdm(ds.subjects):
    #         ras2mni = np.loadtxt(ds.ds_dir / f"{ds.name}.{subject}.mni152_affine_backward.txt")

    #         # image
    #         t1w = nib.load(ds.ds_dir / f"{ds.name}.{subject}.T1w.defacingmask.nii")
    #         vox2ras = t1w.affine
    #         t1w_trans = nib.Nifti1Image(
    #             t1w.get_fdata().squeeze().astype(t1w.get_data_dtype()),
    #             ras2mni @ vox2ras
    #         )
    #         out = nibabel.processing.resample_from_to(t1w_trans, MNI152)
    #         out.to_filename(out_dir / ".".join([ds.name, subject, "T1w.areg-mni", "nii"]))

    #         try:
    #             t1w_mask = nib.load(ds.ds_dir / f"{ds.name}.{subject}.T1w.defacingmask.nii")
    #             t1w_mask_trans = nib.Nifti1Image(
    #                 t1w_mask.get_fdata().squeeze().astype(t1w_mask.get_data_dtype()),
    #                 ras2mni @ vox2ras
    #             )
    #             out = nibabel.processing.resample_from_to(t1w_mask_trans, MNI152)
    #             out.to_filename(out_dir / ".".join([ds.name, subject, "T1w.areg-mni.defacingmask", "nii"]))
    #         except FileNotFoundError:
    #             continue

    #         # surface

    #         affine = torch.tensor(mni_ras2vox @ ras2mni @ vox2ras).float()

    #         d = ".".join([ds.name, subject, "surf_dir"])
    #         if not (out_dir / d).exists():
    #             (out_dir/d).mkdir()

    #         for h in ["lh", "rh"]:
    #             for s in ["white", "pial"]:
    #                 name = ".".join([h,s,"5","target","pt"])

    #                 data = torch.load(ds.ds_dir / d / name)
    #                 out = data @ affine[:3,:3].T + affine[:3,3]
    #                 torch.save(out, out_dir / d / name)
