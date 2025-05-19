from brainsynth.config import DatasetConfig, SynthesizerConfig
from brainsynth.dataset import SynthesizedDataset, setup_dataloader
from brainsynth.synthesizer import Synthesizer

from brainnet.mesh.topology import get_recursively_subdivided_topology

from pathlib import Path
import torch
import nibabel as nib
import numpy as np

top = get_recursively_subdivided_topology(5)


if __name__ == "__main__":

    out = Path("/home/jesperdn/nobackup")

    # SYNTHESIZER
    synthconf = SynthesizerConfig(
        "XSubSynthBuilder",
        device="cuda:0",
        selectable_images=("t1w", "t2w"),
        out_center_str="brain",
        segmentation_labels="brainseg",
    )
    self = Synthesizer(synthconf)

    # # DATASET / DATALOADER
    # dsconf = DatasetConfig(
    #     "/mnt/projects/CORTECH/nobackup/training_data",
    #     "/mnt/projects/CORTECH/nobackup/training_data_subjects",
    #     subject_subset="train",
    #     images = ["generation_labels", "brainseg", "t1w"]
    #     # synthesizer = self,
    # )
    # ds = SynthesizedDataset(
    #     **dsconf.dataset_kwargs["AIBL"]
    # )
    # images, surfaces, initial_vertices = ds[1]

    dsconf = DatasetConfig(
        "/mnt/projects/CORTECH/nobackup/training_data",
        "/mnt/projects/CORTECH/nobackup/training_data_subjects",
        subject_subset="MNI1",
        images = ["generation_labels", "brainseg", "t1w", "mni152_nonlin_forward", "mni152_nonlin_backward"]
        # synthesizer = self,
    )
    ds1 = SynthesizedDataset(
        **dsconf.dataset_kwargs["ABIDE"]
    )

    dsconf = DatasetConfig(
        "/mnt/projects/CORTECH/nobackup/training_data",
        "/mnt/projects/CORTECH/nobackup/training_data_subjects",
        subject_subset="MNI2",
        images = ["mni152_nonlin_forward", "mni152_nonlin_backward"],
        # synthesizer = self,
        target_surface_resolution=None,
    )
    ds2 = SynthesizedDataset(
        **dsconf.dataset_kwargs["ABIDE"]
    )

    i = 4
    images, surfaces, initial_vertices = ds1[0]
    images2, _, _ = ds2[i]
    for k,v in images2.items():
        images[f"other_{k}"] = v

    a, b, c = self(images, surfaces, initial_vertices)

    # dl = setup_dataloader(dsconf.dataset_kwargs) # **dataloader_kwargs

    # images = {k:v[..., :-1] for k,v in images.items()}
    # a, b, c = self(images, surfaces, initial_vertices)

    affine = np.eye(4)

    nib.Nifti1Image(a["t1w"][0].cpu().numpy(), affine).to_filename(out / f"T1w_{i:02d}.nii")
    nib.Nifti1Image(
        a["brainseg"].argmax(0).to(torch.int32).cpu().numpy(), affine
    ).to_filename(out / f"brainseg_{i:02d}.nii")
    nib.Nifti1Image(a["image"][0].cpu().numpy(), affine).to_filename(out / f"image_{i:02d}.nii")

    metadata = dict(
        head=[2, 0, 20],
        valid="1  # volume info valid",
        filename="vol.nii",
        volume=[192, 192, 192],
        voxelsize=[1, 1, 1],
        xras=[-1, 0, 0],
        yras=[0, 0, -1],
        zras=[0, 1, 0],
        cras=[0, 0, 0],
    )
    # b["lh"]["white"][:, 2] -= 0.5
    nib.freesurfer.write_geometry(
        out / f"lh.white_{i:02d}",
        b["lh"]["white"].cpu().numpy(),
        top[5].faces.numpy(),
        volume_info=metadata,
    )
    nib.freesurfer.write_geometry(
        out / f"lh.pial_{i:02d}",
        b["lh"]["pial"].cpu().numpy(),
        top[5].faces.numpy(),
        volume_info=metadata,
    )
    nib.freesurfer.write_geometry(
        out / f"lh.init_{i:02d}",
        c["lh"].cpu().numpy(),
        top[0].faces.numpy(),
        volume_info=metadata,
    )
    # nib.Nifti1Image(
    #     images["T1w"][0].cpu().numpy(), affine
    # ).to_filename(out / "T1w_orig.nii")


# OASIS3    z
# ISBI2015  z
# ADHD200   x
