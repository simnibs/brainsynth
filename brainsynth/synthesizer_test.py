from brainsynth.config import DatasetConfig, SynthesizerConfig
from brainsynth.dataset import SynthesizedDataset, setup_dataloader
from brainsynth.synthesizer import Synthesizer

from brainnet.mesh.topology import get_recursively_subdivided_topology

from pathlib import Path
import torch
import nibabel as nib
import numpy as np

top = get_recursively_subdivided_topology(5)

"""
alias reg_aladin=/mnt/depot64/niftyreg/bin/reg_aladin
alias reg_f3d=/mnt/depot64/niftyreg/bin/reg_f3d
alias reg_transform=/mnt/depot64/niftyreg/bin/reg_transform
alias reg_tools=/mnt/depot64/niftyreg/bin/reg_tools
alias reg_resample=/mnt/depot64/niftyreg/bin/reg_resample

OUT=/home/jesperdn/nobackup/sub2mni_reg_smoo_less
SUB1=/mnt/projects/CORTECH/nobackup/training_data/ABIDE.sub-0001.T1w.nii
SUB2=/mnt/projects/CORTECH/nobackup/training_data/HCP.sub-001.T1w.nii
MNI=/mrhome/jesperdn/nobackup/MNI152_T1_1mm.nii.gz # from fsl

RESAMP_AFF1=$OUT/sub1_resampled_affine.nii
RESAMP_AFF2=$OUT/sub2_resampled_affine.nii
RESAMP_NONLIN1=$OUT/sub1_resampled_nonlin.nii
RESAMP_NONLIN2=$OUT/sub2_resampled_nonlin.nii
AFFINE1=$OUT/affine_sub1_to_mni.txt
AFFINE2=$OUT/affine_sub2_to_mni.txt
AFFINE_INV1=$OUT/affine_mni_to_sub1.txt
AFFINE_INV2=$OUT/affine_mni_to_sub2.txt
CPPVEL1=$OUT/cpp_sub1.nii
CPPVEL2=$OUT/cpp_sub2.nii
CPPVELBACK1=$OUT/cpp_sub1_backward.nii
CPPVELBACK2=$OUT/cpp_sub2_backward.nii
DEFORM1=$OUT/deformation_sub1.nii
DEFORM2=$OUT/deformation_sub2.nii
DEFORM_BACK1=$OUT/deformation_backward_sub1.nii
DEFORM_BACK2=$OUT/deformation_backward_sub2.nii

reg_transform -ref $MNI -ref2 $SUB2 -comp $CPPVEL1 $CPPVELBACK2 $OUT/composed-sub1-sub2.nii
reg_transform -comp $DEFORM1 $DEFORM_BACK2 $OUT/composed-sub1-sub2.nii

reg_transform -comp $DEFORM_BACK2 $DEFORM1 $OUT/composed-sub2-sub1.nii


reg_resample -ref $SUB2 -flo $SUB1 -trans $OUT/composed-sub1-sub2.nii -res $OUT/res.nii

# noAppJL
# WEIGHTS="-be 0.01 -le 0.1 -jl 0.1"
WEIGHTS="-be 0.005 -le 0.05 -jl 0.05"

reg_aladin -ref $MNI -flo $SUB1 -res $RESAMP_AFF1 -aff $AFFINE1
reg_f3d -ref $MNI -flo $SUB1 -vel -cpp $CPPVEL1 -res $RESAMP_NONLIN1 -aff $AFFINE1 $WEIGHTS
reg_transform -invAff $AFFINE1 $AFFINE_INV1
reg_transform -ref $MNI -def $CPPVEL1 $DEFORM1
reg_transform -ref $SUB1 -def $CPPVELBACK1 $DEFORM_BACK1

reg_aladin -ref $MNI -flo $SUB2 -res $RESAMP_AFF2 -aff $AFFINE2
reg_f3d -ref $MNI -flo $SUB2 -vel -cpp $CPPVEL2 -res $RESAMP_NONLIN2 -aff $AFFINE2 $WEIGHTS
reg_transform -invAff $AFFINE2 $AFFINE_INV2
reg_transform -ref $MNI -def $CPPVEL2 $DEFORM2
reg_transform -ref $SUB2 -def $CPPVELBACK2 $DEFORM_BACK2

"""

mni = nib.load("/mrhome/jesperdn/repositories/simnibs-develop/simnibs/resources/templates/MNI152_T1_1mm.nii.gz")
sub2img = nib.load("/mnt/projects/CORTECH/nobackup/training_data/HCP.sub-001.T1w.nii")

img = nib.load("/mnt/projects/CORTECH/nobackup/training_data/ABIDE.sub-0001.T1w.nii")
imgdata = torch.tensor(img.get_fdata()).permute((3,0,1,2)).float()

imgaffine = torch.tensor(img.affine).float()
imgaffine_inv = torch.linalg.inv(imgaffine).float()

surface = torch.load("/mnt/projects/CORTECH/nobackup/training_data/ABIDE.sub-0001.surf_dir/lh.white.5.target.pt")
pial = torch.load("/mnt/projects/CORTECH/nobackup/training_data/ABIDE.sub-0001.surf_dir/lh.pial.5.target.pt")

deform1 = nib.load("/home/jesperdn/nobackup/sub2mni_reg_smoo_less/deformation_sub1.nii")
deform2 = nib.load("/home/jesperdn/nobackup/sub2mni_reg_smoo_less/deformation_sub2.nii")
deform1data = torch.tensor(deform1.get_fdata().squeeze()).float()
deform2data = torch.tensor(deform2.get_fdata().squeeze()).float()

deform_back1 = nib.load("/home/jesperdn/nobackup/sub2mni_reg_smoo_less/deformation_backward_sub1.nii")
deform_back2 = nib.load("/home/jesperdn/nobackup/sub2mni_reg_smoo_less/deformation_backward_sub2.nii")
deform_back1data = torch.tensor(deform_back1.get_fdata().squeeze()).float()
deform_back2data = torch.tensor(deform_back2.get_fdata().squeeze()).float()

# disp1 = nib.load("/home/jesperdn/nobackup/sub2mni_reg_smoo_less/displacement_sub1.nii")
# disp2 = nib.load("/home/jesperdn/nobackup/sub2mni_reg_smoo_less/displacement_sub2.nii")
# disp1data = torch.tensor(disp1.get_fdata().squeeze()).float()
# disp2data = torch.tensor(disp2.get_fdata().squeeze()).float()

# disp_back1 = nib.load("/home/jesperdn/nobackup/sub2mni_reg_smoo_less/displacement_backward_sub1.nii")
# disp_back2 = nib.load("/home/jesperdn/nobackup/sub2mni_reg_smoo_less/displacement_backward_sub2.nii")
# disp_back1data = torch.tensor(disp_back1.get_fdata().squeeze()).float()
# disp_back2data = torch.tensor(disp_back2.get_fdata().squeeze()).float()

mniaffine = torch.tensor(deform2.affine).float()
mniaffine_inv = torch.linalg.inv(mniaffine)

from brainsynth.transforms.spatial import GridSample

composed = nib.load("/home/jesperdn/nobackup/sub2mni_reg_smoo_less/composed-sub1-sub2.nii")
composeddata = torch.tensor(composed.get_fdata().squeeze()).float()

# for each vox in MNI what is the corresponding vox in sub1
grid = composeddata @ imgaffine_inv[:3,:3] + imgaffine_inv[:3,3]
# sample those positions...
sampler = GridSample(grid, imgdata.shape[-3:])
imgdata_sub2 = sampler(imgdata)

nib.Nifti1Image(imgdata_sub2.float().numpy().squeeze(),
                composed.affine).to_filename(
    "/home/jesperdn/nobackup/sub2mni_reg_smoo_less/sub1-to-sub2-direct.nii"
)



    nib.Nifti1Image(imgdata_mni.float().numpy().squeeze(),
                    sub2img.affine).to_filename(
        "/home/jesperdn/nobackup/sub2mni_reg_smoo_less/sub1-to-sub2-direct1.nii"
    )


#transform IMAGE from SUB1 space to MNI space:

# for each vox in MNI what is the corresponding vox in sub1
grid = deform1data @ imgaffine_inv[:3,:3] + imgaffine_inv[:3,3]
# sample those positions...
sampler = GridSample(grid, imgdata.shape[-3:])
imgdata_mni = sampler(imgdata)


#transform IMAGE_MNI from MNI space to SUB2 space:

grid = deform_back2data @ mniaffine_inv[:3, :3].T + mniaffine_inv[:3, 3]
sampler = GridSample(grid, imgdata_mni.shape[-3:])
imgdata_sub2 = sampler(imgdata_mni)



nib.Nifti1Image(imgdata_mni.float().numpy().squeeze(),
                deform1.affine).to_filename(
    "/home/jesperdn/nobackup/sub2mni_reg_smoo_less/sub1-to-mni.nii"
)
nib.Nifti1Image(imgdata_sub2.float().numpy().squeeze(),
                deform_back2.affine).to_filename(
    "/home/jesperdn/nobackup/sub2mni_reg_smoo_less/sub1-to-sub2.nii"
)

#transform SURFACE from SUB1 space to MNI space:

grid = surface
sample = GridSample(grid, deform_back1data.shape[:3])
surface_mni = sample(deform_back1data.permute((3,0,1,2))).squeeze().T

#transform SURFACE_MNI from MNI space to SUB2 space:

# to MNI voxel space
grid = surface_mni @ mniaffine_inv[:3, :3].T + mniaffine_inv[:3, 3]
sample = GridSample(grid, deform2data.shape[:3])
surface_sub2 = sample(deform2data.permute((3,0,1,2))).squeeze().T

# PIAL

grid = pial
sample = GridSample(grid, deform_back1data.shape[:3])
pial_mni = sample(deform_back1data.permute((3,0,1,2))).squeeze().T

#transform SURFACE_MNI from MNI space to SUB2 space:

# to MNI voxel space
grid = pial_mni @ mniaffine_inv[:3, :3].T + mniaffine_inv[:3, 3]
sample = GridSample(grid, deform2data.shape[:3])
pial_sub2 = sample(deform2data.permute((3,0,1,2))).squeeze().T



metadata = dict(
    head=[2, 0, 20],
    valid="1  # volume info valid",
    filename="vol.nii",
    volume=[182, 218, 182],
    voxelsize=[1, 1, 1],
    xras=[-1, 0, 0],
    yras=[0, 0, -1],
    zras=[0, 1, 0],
    cras=[0, 0, 0],
)

nib.freesurfer.write_geometry(
    "/home/jesperdn/nobackup/sub2mni_reg_smoo_less/lh.white.sub1-to-mni",
    surface_mni.float().numpy(), top[5].faces.numpy(), volume_info=metadata)
nib.freesurfer.write_geometry(
    "/home/jesperdn/nobackup/sub2mni_reg_smoo_less/lh.pial.sub1-to-mni",
    pial_mni.float().numpy(), top[5].faces.numpy(), volume_info=metadata)

metadata = dict(
    head=[2, 0, 20],
    valid="1  # volume info valid",
    filename="vol.nii",
    volume=[182, 218, 182],
    voxelsize=[1, 1, 1],
    xras=[-1, 0, 0],
    yras=[0, 0, -1],
    zras=[0, 1, 0],
    cras=[0, 0, 0],
)

nib.freesurfer.write_geometry(
    "/home/jesperdn/nobackup/sub2mni_reg_smoo_less/lh.white.sub1-to-sub2",
    surface_sub2.float().numpy(), top[5].faces.numpy(), volume_info=metadata)
nib.freesurfer.write_geometry(
    "/home/jesperdn/nobackup/sub2mni_reg_smoo_less/lh.pial.sub1-to-sub2",
    pial_sub2.float().numpy(), top[5].faces.numpy(), volume_info=metadata)


reference   The stationary images
floating    The moving image

Deformation

The value in each voxel (reference image space) gives the RAS coordinate in the
floating image corresponding to this voxel.

Displacement

The value in each voxel (reference image space) gives the difference between
the RAS coordinate of that voxel and the RAS coordinate of the corresponding
location in the floating image, i.e.,

    RAS[REF][i,j,k] + DISP[i,j,k] = DEFORM[i,j,k] = RAS[FLO] (of corresponding point)



if __name__ == "__main__":

    out = Path("/home/jesperdn/nobackup")

    # SYNTHESIZER
    synthconf = SynthesizerConfig(
        device="cuda:0",
        alternative_images=("t1w", "t2w"),
        # out_center_str="lh",
        segmentation_labels="brainseg",
    )
    self = Synthesizer(synthconf)

    # DATASET / DATALOADER
    dsconf = DatasetConfig(
        "/mnt/projects/CORTECH/nobackup/training_data",
        "/mnt/projects/CORTECH/nobackup/training_data_subjects",
        subject_subset="train",
        images = ["generation_labels", "brainseg", "t1w"]
        # synthesizer = self,
    )
    ds = SynthesizedDataset(
        **dsconf.dataset_kwargs["AIBL"]
    )
    images, surfaces, initial_vertices = ds[1]

    # dl = setup_dataloader(dsconf.dataset_kwargs) # **dataloader_kwargs

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
