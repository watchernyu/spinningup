#!/bin/sh
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB

#SBATCH --array=0-59
##SBATCH --output=sbl_%A_%a.out #if you need each subjob to generate an output file.
#SBATCH --output=sbl_%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID, which is 0-9

#SBATCH --constraint=cpu

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load anaconda3 gcc/7.3
source activate rl

echo ${SLURM_ARRAY_TASK_ID}
python hpc_mujoco_job_array.py --setting ${SLURM_ARRAY_TASK_ID}
