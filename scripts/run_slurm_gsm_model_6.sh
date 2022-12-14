#!/bin/bash

#SBATCH --requeue                  # Return job to the queue if preempted
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=gsm_model_6              # Assign an short name to your job
#SBATCH --cpus-per-task=2          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=32GB                 # Real memory (RAM) required
#SBATCH --gres=gpu:rtx8000:2               # Generic resources
#SBATCH --time=10:00:00            # Total run time limit (HH:MM:SS)
#SBATCH --error=%x.err
#SBATCH --output=%x.out
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=cg4177@nyu.edu # send me mail

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate idls_project_1

python /home/cg4177/t5-tuning/t5_gsm8k.py --identifier model_6 --model_name t5-large --batch_size 4 --epochs 20 --fp_precision 32 --devices 2 --num_workers 2 --strategy ddp_sharded