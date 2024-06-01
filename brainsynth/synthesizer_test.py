from brainsynth.config.augmentation import AugmentationConfig
from brainsynth.dataset import AugmentedDataset
from brainsynth.synthesizer import Synthesizer

from brainnet.mesh.topology import get_recursively_subdivided_topology

from pathlib import Path
import torch
import nibabel as nib
import numpy as np

top = get_recursively_subdivided_topology(5)


if __name__ == "__main__":

    out = Path("/home/jesperdn/nobackup")

    config = AugmentationConfig(
        device="cuda:0",
        alternative_images=("t1w", "t2w"),
        # out_center_str="lh",
        segmentation_labels="brainseg",
    )
    self = Synthesizer(config)

    ds = AugmentedDataset(
        ds_dir="/mnt/projects/CORTECH/nobackup/training_data",
        ds_name="ADHD200",
        subjects=["sub-001", "sub-010"],
        synthesizer=None,
        images=["generation_labels", "brainseg", "t1w"],
        ds_structure="flat",
        target_surface_resolution=5,
        target_surface_hemispheres="lh",
        initial_surface_resolution=0,
    )
    images, surfaces, initial_vertices = ds[1]

    # images = {k:v[..., :-1] for k,v in images.items()}
    a, b, c = self(images, surfaces, initial_vertices)

    affine = np.eye(4)

    nib.Nifti1Image(a["t1w"][0].cpu().numpy(), affine).to_filename(out / "T1w.nii")
    nib.Nifti1Image(
        a["brainseg"].argmax(0).to(torch.int32).cpu().numpy(), affine
    ).to_filename(out / "brainseg.nii")
    nib.Nifti1Image(a["image"][0].cpu().numpy(), affine).to_filename(out / "image.nii")

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
        out / "lh.white",
        b["lh"]["white"].cpu().numpy(),
        top[5].faces.numpy(),
        volume_info=metadata,
    )
    nib.freesurfer.write_geometry(
        out / "lh.pial",
        b["lh"]["pial"].cpu().numpy(),
        top[5].faces.numpy(),
        volume_info=metadata,
    )
    # nib.Nifti1Image(
    #     images["T1w"][0].cpu().numpy(), affine
    # ).to_filename(out / "T1w_orig.nii")


# OASIS3    z
# ISBI2015  z
# ADHD200   x
