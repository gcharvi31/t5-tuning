#!/bin/bash

#SBATCH --requeue                  # Return job to the queue if preempted
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=gsm_model_7              # Assign an short name to your job
#SBATCH --cpus-per-task=4          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=32GB                 # Real memory (RAM) required
#SBATCH --gres=gpu:v100:4               # Generic resources
#SBATCH --time=12:00:00            # Total run time limit (HH:MM:SS)
#SBATCH --error=%x.err
#SBATCH --output=%x.out
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=cg4177@nyu.edu # send me mail

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate idls_project_1

python /home/cg4177/t5-tuning/t5_gsm8k_v1.py --identifier model_7 --model_name t5-3b --batch_size 4 --epochs 10 --fp_precision 16 --devices 4 --num_workers 4 --strategy ddp_sharded