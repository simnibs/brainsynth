#!/bin/bash

#SBATCH --job-name=reconall         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=8           # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=24G                   # Job memory request
#SBATCH --array=1-581           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279
#SBATCH --exclude=big20,big21,big27,big28,small1,small2,small19,small23,small24,small25,small29,small30,small31,small32,small33,small34,small35,drakkisath,rivendare

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


# get list of the failed runs...
# rg terminated . -l | cut -d "_" -f 3 | cut -d "." -f 1 | sort | paste -sd ","


# module load freesurfer/7.4.1
module load freesurfer/8.0.0

DATASET_DIR=$CORTECH/nobackup/jesper/IXI

SUB_T1=$(find $DATASET_DIR/T1 -type f -name "IXI*" | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
SUB_ID=$(basename $SUB_T1)
SUB_ID=${SUB_ID:0:6} # keep IXI***

SUB_T2=$(find $DATASET_DIR/T2 -type f -name "$SUB_ID*.nii.gz")

echo "Using T1 : $SUB_T1"
echo "Using T2 : $SUB_T2"

export SUBJECTS_DIR=$SCRATCH/IXI_FS

# modify recon-all script to run synthstrip with --no-csf
# -noaseg
# $SCRATCH/datasets/recon-all-xopts -s $SUB_ID -i $SUB_T1 -all -threads 8
# $SCRATCH/datasets/recon-all-xopts -s $SUB_ID -autorecon3 -threads 8

recon-all -s $SUB_ID -i $SUB_T1 -all

# if this subject has T2, use it, and ensure that new statistics are computed
# if [ -n $SUB_T2 ]; then
#     recon-all -s $SUB_ID -T2 $SUB_T2 -T2pial -autorecon3
# fi

# remove failed runs...
# for i in $(rg terminated $SCRATCH/slurm_logs -l | cut -d "_" -f 4 | cut -d "." -f 1 | sort | paste -sd " ")
# do
#     SUB_DIR=$(find $SCRATCH/IXI_FS/ -maxdepth 1 -type d -regex '.*IXI[0-9][0-9][0-9]'| sort | sed -n "${i}p")
#     echo remove $SUB_DIR
#     rm -r $SUB_DIR
# done

# TO_REMOVE=$(find $SCRATCH/IXI_FS/ -name recon-all.log -exec rg terminated -l {} + | xargs dirname | xargs dirname)
# for i in $TO_REMOVE; do
#     rm -r $i
# done

