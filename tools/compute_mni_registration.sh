#!/bin/bash

#SBATCH --job-name=niftyreg          # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=4           # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=4G                   # Job memory request
#SBATCH --array=10-5279%100           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279

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

# Niftyreg Terminology
# --------------------
# Deformation (this is what we use here)
#
# The value in each voxel (reference image space) gives the RAS coordinate in the
# floating image corresponding to this voxel.
#
# Displacement
#
# The value in each voxel (reference image space) gives the difference between
# the RAS coordinate of that voxel and the RAS coordinate of the corresponding
# location in the floating image, i.e.,
#
#     RAS[REF][i,j,k] + DISP[i,j,k] = DEFORM[i,j,k] = RAS[FLO] (of corresponding point)


# By default, functions are not exported to be available in subshells so we
# need this before we can use 'conda activate'
source ~/mambaforge/etc/profile.d/conda.sh
conda activate synth

export PATH=/mnt/depot64/niftyreg/bin:$PATH

OUT=/mnt/scratch/personal/jesperdn/training_data_deformations
# We use this MNI152 template from FSL as our "common" space
MNI152=/mnt/depot64/fsl/fsl.6.0.5/data/standard/MNI152_T1_1mm.nii.gz
# contains full path to all subject T1w images
SUB_T1w=$(sed -n $SLURM_ARRAY_TASK_ID"p" /mnt/projects/CORTECH/nobackup/ALL_T1w.txt)
# extract the dataset + subject prefix
SUB_PREFIX=$OUT/$(python -c "from pathlib import Path; p = Path('$SUB_T1w'); ps = p.name.split('.'); print('.'.join(ps[:2]))")

AFFINE="${SUB_PREFIX}.mni152_affine.txt"
AFFINE_BACK="${SUB_PREFIX}.mni152_affine_backward.txt"
AFFINE_FORWARD="${SUB_PREFIX}.mni152_affine_forward.txt"
RESAMP_AFF="${SUB_PREFIX}.resamp_affine.nii"
RESAMP_NONLIN="${SUB_PREFIX}.resamp_nonlin.nii"
RESAMP_NONLIN_BACK="${SUB_PREFIX}.resamp_nonlin_backward.nii"

CPP="${SUB_PREFIX}.cpp.nii"
CPP_BACK="${SUB_PREFIX}.cpp_backward.nii"
MNIREG="${SUB_PREFIX}.mni152_nonlin.nii"
MNIREG_BACK="${SUB_PREFIX}.mni152_nonlin_backward.nii"
MNIREG_FORWARD="${SUB_PREFIX}.mni152_nonlin_forward.nii"

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
reg_transform -ref $MNI152 -def $CPP $MNIREG
reg_transform -ref $SUB_T1w -def $CPP_BACK $MNIREG_BACK

# convert registrations from world coordinates to voxel coordinates, multiply by 100, and cast to int16
python -c "
import nibabel as nib;
from nibabel.affines import apply_affine;
import numpy as np;
mni152_nonlin = nib.load('$MNIREG');
mni152_nonlin_back = nib.load('$MNIREG_BACK');
data = np.round(100 * apply_affine(np.linalg.inv(mni152_nonlin_back.affine), mni152_nonlin.get_fdata().squeeze()));
out = nib.Nifti1Image(data.astype(np.int16), mni152_nonlin.affine);
out.to_filename('$MNIREG');
data = np.round(100 * apply_affine(np.linalg.inv(mni152_nonlin.affine), mni152_nonlin_back.get_fdata().squeeze()));
out = nib.Nifti1Image(data.astype(np.int16), mni152_nonlin_back.affine);
out.to_filename('$MNIREG_BACK');
"

mv $MNIREG $MNIREG_FORWARD
mv $AFFINE $AFFINE_FORWARD

rm $RESAMP_AFF $RESAMP_NONLIN $RESAMP_NONLIN_BACK $CPP $CPP_BACK
