#!/bin/sh
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB

# use this command outside a comment to request for one gpu: #SBATCH --gres=gpu:1

#SBATCH --array=0-9
##SBATCH --output=blhum_%A_%a.out #if you need each subjob to generate an output file.
#SBATCH --output=blhum_%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID, which is 0-9
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load anaconda3 gcc/7.3
source activate rl

echo ${SLURM_ARRAY_TASK_ID}
python sample_job_array_grid.py --env HalfCheetah-v2 --seed ${SLURM_ARRAY_TASK_ID}
