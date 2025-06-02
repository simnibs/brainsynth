#!/bin/bash

#SBATCH --job-name=resample         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=1          # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=2G                   # Job memory request
#SBATCH --array=1-200           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279


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

ROOT_SOURCE=$CORTECH/nobackup/jesper/evaluation_data/ADNI-GO2
ROOT_DEST=$SCRATCH/ADNI-GO2

SRC_SUB_DIR=$(find $ROOT_SOURCE -maxdepth 2 -type d -name "sub-*" | sed -n "${SLURM_ARRAY_TASK_ID}p")
DST_SUB_DIR=${SRC_SUB_DIR/${ROOT_SOURCE}/${ROOT_DEST}}

echo $SRC_SUB_DIR
echo $DST_SUB_DIR
mkdir -p $DST_SUB_DIR

# find /mnt/projects/CORTECH/nobackup/training_data/full/ -maxdepth 2 -type d -name "sub-*" > $CORTECH/nobackup/training_data/full_all_subjects.txt
#ALL_SUBJECTS=/mnt/projects/CORTECH/nobackup/training_data/full_all_subjects.txt
#subject_path=$(cat $ALL_SUBJECTS | sed -n "${SLURM_ARRAY_TASK_ID}p")

SCRIPT="${repo}/tools/resample_surfaces.py"
cmd="python $SCRIPT ${SRC_SUB_DIR} ${DST_SUB_DIR}"

echo
echo "Executing : $cmd"
echo

$cmd


# find /mnt/projects/CORTECH/nobackup/training_data/full/ -type f -name *h.*.target.pt
# -delete

# find /mnt/projects/CORTECH/nobackup/training_data/full/ -type f -name *h.0.template.pt
# -delete

# SOURCE=/mnt/scratch/personal/jesperdn/surface_data/
# DEST=/mnt/projects/CORTECH/nobackup/training_data/full/
# rsync -auvhn $SOURCE $DEST