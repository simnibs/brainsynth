from pathlib import Path
import sys

import nibabel as nib
import nibabel.processing
import numpy as np
# from tqdm import tqdm

if __name__ == "__main__":


    index = int(sys.argv[1])

    do_surface = True
    do_t1w = False
    do_brainseg_with_extracerebral = False
    do_generation_labels_dist = False

    out_dir = Path("/mnt/scratch/personal/jesperdn/training_data/brainreg")

    MNI152 = nib.load("/home/jesperdn/repositories/brainnet/MNI152_T1_1mm_192_224_192.nii.gz")

    import brainsynth
    from brainsynth.config.dataset import DatasetConfig
    import torch

    root_dir = Path("/mnt/projects/CORTECH/nobackup")
    ds_conf = DatasetConfig(
        root_dir = root_dir / "training_data" / "full",
        subject_dir = root_dir / "training_data" / "subject_splits",
        subject_subset=None,
        images=[],
        target_surface_resolution=5,
        target_surface_hemispheres="both",
        initial_surface_resolution=0,
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

    sub_in_dir = ds.ds_dir / ds.name / subject
    sub_out_dir = out_dir / ds.name / subject
    if not sub_out_dir.exists():
        sub_out_dir.mkdir(parents=True)

    ras2mni = np.loadtxt(sub_in_dir / "mni152_affine_backward.txt")

    if do_t1w:
        t1w = nib.load(sub_in_dir / "T1w.nii")
        vox2ras = t1w.affine
        t1w_trans = nib.Nifti1Image(
            t1w.get_fdata().squeeze().astype(t1w.get_data_dtype()),
            ras2mni @ vox2ras
        )
        out = nibabel.processing.resample_from_to(t1w_trans, MNI152)
        out.to_filename(sub_out_dir / "T1w.areg-mni.nii")


        try:
            t1w_mask = nib.load(sub_in_dir / "T1w.defacingmask.nii")
            t1w_mask_trans = nib.Nifti1Image(
                t1w_mask.get_fdata().squeeze().astype(t1w_mask.get_data_dtype()),
                ras2mni @ vox2ras
            )
            out = nibabel.processing.resample_from_to(t1w_mask_trans, MNI152)
            out.to_filename(sub_out_dir / "T1w.areg-mni.defacingmask.nii")
        except FileNotFoundError:
            pass


    if do_brainseg_with_extracerebral:
        seg = nib.load(sub_in_dir / "brainseg_with_extracerebral.nii")
        vox2ras = seg.affine
        seg_trans = nib.Nifti1Image(
            seg.get_fdata().squeeze().astype(seg.get_data_dtype()),
            ras2mni @ vox2ras
        )
        out = nibabel.processing.resample_from_to(seg_trans, MNI152, order=0)
        out.to_filename(sub_out_dir / "brainseg_with_extracerebral.nii")

    if do_generation_labels_dist:
        gen = nib.load(sub_in_dir / "generation_labels_dist.nii")
        vox2ras = gen.affine
        gen_trans = nib.Nifti1Image(
            gen.get_fdata().squeeze().astype(gen.get_data_dtype()),
            ras2mni @ vox2ras
        )
        out = nibabel.processing.resample_from_to(gen_trans, MNI152, order=0)
        out.to_filename(sub_out_dir / "generation_labels_dist.nii")


    if do_surface:
        # we need vox2ras!
        t1w = nib.load(sub_in_dir/ "T1w.nii")
        vox2ras = t1w.affine

        affine = torch.tensor(mni_ras2vox @ ras2mni @ vox2ras).float()

        for h in ["lh", "rh"]:
            # template
            name = ".".join([h,"0","template","pt"])

            data = torch.load(sub_in_dir / name)
            out = data @ affine[:3,:3].T + affine[:3,3]
            torch.save(out, sub_out_dir / name)

            # target
            for s in ("white", "pial"):

                name = ".".join([h,s,"5","target-decoupled","pt"])

                data = torch.load(sub_in_dir / name)
                out = data @ affine[:3,:3].T + affine[:3,3]
                torch.save(out, sub_out_dir / name)

