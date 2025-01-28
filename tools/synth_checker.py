from pathlib import Path

import nibabel as nib
import numpy as np
import torch

import brainsynth
import brainsynth.config
import brainsynth.dataset

import brainnet.mesh.topology


n_examples = 20
out_dir = Path("/home/jesperdn/nobackup/synthexamples")

dataset_n = 0
subject_n = 0

root_dir = Path("/mnt/projects/CORTECH/nobackup/")
device = torch.device("cuda:0")
initial_surface_resolution: int = 0
resolution = 6
target_surface = dict(resolution=resolution, name="target-decoupled")
out_size = [176, 208, 176]
out_center_str = "brain"


def write_surface(surfaces, vol_info, faces, filename):
    for hemi, s in surfaces.items():
        for surf, v in s.items():
            nib.freesurfer.write_geometry(
                filename.with_name(filename.name + f"_{hemi}-{surf}"),
                v.detach().to(torch.float).cpu().numpy(),
                faces,
                volume_info=vol_info,
            )

def write_init(surfaces, vol_info, faces, filename):
    for hemi, v in surfaces.items():
        nib.freesurfer.write_geometry(
            filename.with_name(filename.name + f"_{hemi}"),
            v.detach().to(torch.float).cpu().numpy(),
            faces,
            volume_info=vol_info,
        )


synth = brainsynth.Synthesizer(
    brainsynth.config.SynthesizerConfig(
        builder="OnlySynthIso",
        out_size=out_size,
        out_center_str=out_center_str,
        alternative_images=["t1w"],  # ["t1w", "t2w", "flair"],
        device=device,
    )
)

topologies = brainnet.mesh.topology.StandardTopology.recursive_subdivision(resolution)
faces_init = topologies[0].faces.numpy()
faces_surf = topologies[-1].faces.numpy()

affine = np.eye(4)

vol_info = dict(
    head=[2, 0, 20],
    valid="1  # volume info valid",
    filename="vol.nii",
    voxelsize=[1, 1, 1],
    volume=(0, 0, 0),
    xras=[-1, 0, 0],
    yras=[0, 0, -1],
    zras=[0, 1, 0],
    cras=[0, 0, 0],
)
vol_info["volume"] = (256, 256, 256)


cfg_dataset = brainsynth.config.DatasetConfig(
    root_dir=root_dir / "training_data" / "full",
    subject_dir=root_dir / "training_data" / "subject_splits",
    subject_subset="validation",
    images = ["generation_labels_dist", "t1w"],
    target_surface=target_surface,
)
dataset = brainsynth.dataset.concatenate_datasets(cfg_dataset)
batch = dataset.datasets[dataset_n][subject_n]
images, surfaces, init_verts = batch

for i in range(n_examples):
    print(i)

    with torch.no_grad():
        y_true = synth(images, surfaces, init_verts, unpack=False)

    img = nib.Nifti1Image(y_true["image"][0].cpu().numpy(), affine)
    img.to_filename(out_dir / f"{i:04d}_synth.nii")

    write_init(
        y_true["initial_vertices"],
        vol_info,
        faces_init,
        out_dir / f"{i:04d}_init",
    )


img = nib.Nifti1Image(y_true["t1w"][0, ...].cpu().numpy(), affine)
img.to_filename(out_dir / "t1w.nii")

write_surface(
    y_true["surface"], vol_info, faces_surf, out_dir / "synth_gt"
)
