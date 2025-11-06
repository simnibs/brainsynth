#!/bin/bash

#SBATCH --job-name=prepare_dataset         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=1          # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=2G                   # Job memory request
#SBATCH --array=1-1238           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279

# Get the number of subjects for each dataset
# for d in $CORTECH/nobackup/training_data/full/*; do
#     N=$(find $d -maxdepth 1 -name "sub-*" | wc -l)
#     echo $(basename $d) : $N
# done

echo "Job Information"
echo
echo "Job name     :  $SLURM_JOB_NAME"
echo "Job ID       :  $SLURM_ARRAY_JOB_ID"
echo "Task ID      :  $SLURM_ARRAY_TASK_ID"
echo "Cluster name :  $SLURM_CLUSTER_NAME"
echo "Node name    :  $SLURM_NODENAME"
echo "Date         :  $(date)"
echo "Working dir  :  $SLURM_SUBMIT_DIR"
echo


# By default, functions are not exported to be available in subshells so we
# need this before we can use 'conda activate'
source ~/mambaforge/etc/profile.d/conda.sh
conda activate synth

repo=/home/jesperdn/repositories/brainsynth

#DATASET=ADNI-GO2
DATASET=OASIS3
if [ $DATASET = IXI ]; then
    ROOT_DEST=$SCRATCH/$DATASET
    ROOT_SOURCE=$CORTECH/nobackup/jesper/$DATASET
    FREESURFER_SOURCE=$ROOT_SOURCE/freesurfer
    SRC_SUB_DIR=$(find $FREESURFER_SOURCE -maxdepth 1 -type d -regex ".*IXI[0-9][0-9][0-9]" | sed -n "${SLURM_ARRAY_TASK_ID}p")
    SRC_SUB_T2=$(find $ROOT_SOURCE/T2/$SUBJECT_ID*)
    SRC_SUB_PD=$(find $ROOT_SOURCE/PD/$SUBJECT_ID*)
else
    ROOT_SOURCE=$CORTECH/nobackup/training_data/full/$DATASET
    ROOT_DEST=$SCRATCH/full/$DATASET
    FREESURFER_SOURCE=$ROOT_SOURCE
    SRC_SUB_DIR=$(find $FREESURFER_SOURCE -maxdepth 1 -type d -name "sub-*" | sed -n "${SLURM_ARRAY_TASK_ID}p")
fi

SUBJECT_ID=$(basename $SRC_SUB_DIR)

DST_SUB_DIR=$ROOT_DEST/$SUBJECT_ID
# DST_SUB_DIR=${SRC_SUB_DIR/${FREESURFER_SOURCE}/${ROOT_DEST}}


echo "INPUT  : $SRC_SUB_DIR"
echo "T2     : $SRC_SUB_T2"
echo "PD     : $SRC_SUB_PD"
echo "OUTPUT : $DST_SUB_DIR"
mkdir -p $DST_SUB_DIR

echo
echo PREPARE T1w
cmd="python ${repo}/tools/prepare_t1w.py $SRC_SUB_DIR $DST_SUB_DIR/T1w.nii"
echo "Executing : $cmd"
$cmd

echo
echo PREPARE T2w
cmd="python ${repo}/tools/prepare_other.py $SRC_SUB_DIR $SRC_SUB_T2 $DST_SUB_DIR/T2w.nii"
echo "Executing : $cmd"
$cmd

echo
echo PREPARE PD
cmd="python ${repo}/tools/prepare_other.py $SRC_SUB_DIR $SRC_SUB_PD $DST_SUB_DIR/PD.nii"
echo "Executing : $cmd"
$cmd

echo
echo COPY SURFACES
for hemi in lh rh; do
    for surface in white pial sphere.reg; do
        cp $SRC_SUB_DIR/surf/$hemi.$surface $DST_SUB_DIR/
    done
done

# cp $SRC_SUB_DIR/T1w.nii $DST_SUB_DIR/

# same in and out dir!
echo
echo PREPARE SURFACES
cmd="python ${repo}/tools/prepare_surfaces.py $DST_SUB_DIR $DST_SUB_DIR"
echo "Executing : $cmd"
$cmd

echo
echo PREPARE AFFINE MNI TRANSFORM
cmd="python ${repo}/tools/prepare_mni2ras_transform.py ${DST_SUB_DIR}"
echo "Executing : $cmd"
$cmd

echo
echo PREPARE TEMPLATE
cmd="python ${repo}/tools/prepare_template.py ${DST_SUB_DIR} ${DST_SUB_DIR}"
echo "Executing : $cmd"
$cmd




# find /mnt/projects/CORTECH/nobackup/training_data/full/ -type f -name *h.*.target.pt
# -delete

# find /mnt/projects/CORTECH/nobackup/training_data/full/ -type f -name *h.0.template.pt
# -delete

# SOURCE=$SCRATCH/full/
# DEST=$CORTECH/nobackup/training_data/full/
# rsync -auvh $SOURCE $DEST -n

# rsync -auvhn --include="/*" --include="*.resample" --exclude="*" $SOURCE $DEST



# SOURCE=$SCRATCH/FS_ADNI-GO2/
# DEST=$CORTECH/nobackup/training_data/full/ADNI-GO2/
# rsync -auvhn --include="/*" --include="*h.white" --exclude="*" $SOURCE $DEST
# rsync -auvhn --include="/*" --include="*h.pial" --exclude="*" $SOURCE $DEST
# rsync -auvhn --include="/*" --include="*h.sphere.reg" --exclude="*" $SOURCE $DEST

# SOURCE=$SCRATCH/FS_ADNI-GO2
# DEST=$CORTECH/nobackup/training_data/full/ADNI-GO2

# cd $SOURCE
# for d in ./*; do for f in $d/surf/*h.white; do echo "cp $f $DEST/$(basename $d)/$(basename $f)"; done; done

# for d in ./*; do for f in $d/surf/*h.pial; do cp $f $DEST/$(basename $d)/$(basename $f); done; done
