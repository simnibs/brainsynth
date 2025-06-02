#!/bin/bash

#SBATCH --job-name=resample         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=1          # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=2G                   # Job memory request
#SBATCH --array=1-5279           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279


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


# find /mnt/projects/CORTECH/nobackup/training_data/full/ -maxdepth 2 -type d -name "sub-*" > $CORTECH/nobackup/training_data/full_all_subjects.txt
ALL_SUBJECTS=/mnt/projects/CORTECH/nobackup/training_data/full_all_subjects.txt
subject_path=$(cat $ALL_SUBJECTS | sed -n "${SLURM_ARRAY_TASK_ID}p")

SCRIPT="${repo}/tools/align_template_to_subject.py"
cmd="python $SCRIPT ${subject_path}"

echo
echo "Executing : $cmd"
echo

$cmd
