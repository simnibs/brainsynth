#!/bin/bash

#SBATCH --job-name=niftyreg          # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=4           # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=4G                   # Job memory request
#SBATCH --array=6-10%50           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279

echo "Job Information"
echo
echo "Job name     :  $SLURM_JOB_NAME"
echo "Job ID       :  $SLURM_ARRAY_JOB_ID"
echo "Task ID      :  $SLURM_ARRAY_TASK_ID"
echo "Cluster name :  $SLURM_CLUSTER_NAME"
echo "Node name    :  $SLURMD_NODENAME"
echo "Date         :  $(date)"
echo "Working dir  :  $SLURM_SUBMIT_DIR"
echo

# Compute registration (affine and nonlinear) to MNI152 space
# forward is sub -> MNI
# backward is MNI -> sub

export PATH=/mnt/depot64/niftyreg/bin:$PATH

OUT=/mnt/scratch/personal/jesperdn/training_data_deformations
# We use this MNI152 template from FSL as our "common" space
MNI152=/mnt/depot64/fsl/fsl.6.0.5/data/standard/MNI152_T1_1mm.nii.gz
# contains full path to all subject T1w images
SUB_T1w=$(sed -n $SLURM_ARRAY_TASK_ID"p" /mnt/projects/CORTECH/nobackup/ALL_T1w.txt)

SUB_PREFIX=$OUT/$(python -c "from pathlib import Path; p = Path('$SUB_T1w'); ps = p.name.split('.'); print('.'.join(ps[:2]))")

AFFINE="${SUB_PREFIX}.affine.txt"
AFFINE_BACK="${SUB_PREFIX}.affine_backward.txt"
AFFINE_FINAL="${SUB_PREFIX}.affine_forward.txt"
RESAMP_AFF="${SUB_PREFIX}.resamp_affine.nii"
RESAMP_NONLIN="${SUB_PREFIX}.resamp_nonlin.nii"
RESAMP_NONLIN_BACK="${SUB_PREFIX}.resamp_nonlin_backward.nii"

CPP="${SUB_PREFIX}.cpp.nii"
CPP_BACK="${SUB_PREFIX}.cpp_backward.nii"
DEFORM="${SUB_PREFIX}.deform.nii"
DEFORM_BACK="${SUB_PREFIX}.deform_backward.nii"
DEFORM_FINAL="${SUB_PREFIX}.deform_forward.nii"

# regularization weights
# WEIGHTS="-be 0.001 -le 0.01"
# WEIGHTS="-be 0.005 -le 0.05 -jl 0.05"
WEIGHTS="-be 0.005 -le 0.05"

# Affine registration
reg_aladin -ref $MNI152 -flo $SUB_T1w -res $RESAMP_AFF -aff $AFFINE

# Nonlinear registration
reg_f3d \
    -ref $MNI152 \
    -flo $SUB_T1w \
    -vel \
    -cpp $CPP \
    -res $RESAMP_NONLIN \
    -aff $AFFINE \
    $WEIGHTS

reg_transform -invAff $AFFINE $AFFINE_BACK
# Write the deformation fields
reg_transform -ref $MNI152 -def $CPP $DEFORM
reg_transform -ref $SUB_T1w -def $CPP_BACK $DEFORM_BACK

# convert deformations from world coordinates to voxel coordinates
python -c "
import nibabel as nib;
from nibabel.affines import apply_affine;
import numpy as np;
deform = nib.load('$DEFORM');
deform_back = nib.load('$DEFORM_BACK');
out = nib.Nifti1Image(apply_affine(np.linalg.inv(deform_back.affine), deform.get_fdata().squeeze()).astype(np.float32), deform.affine);
out.to_filename('$DEFORM');
out = nib.Nifti1Image(apply_affine(np.linalg.inv(deform.affine), deform_back.get_fdata().squeeze()).astype(np.float32), deform_back.affine);
out.to_filename('$DEFORM_BACK');
"

mv $DEFORM $DEFORM_FINAL
mv $AFFINE $AFFINE_FINAL

rm $RESAMP_AFF $RESAMP_NONLIN $RESAMP_NONLIN_BACK $CPP $CPP_BACK
