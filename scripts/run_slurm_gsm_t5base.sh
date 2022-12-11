#!/bin/bash

#SBATCH --requeue                  # Return job to the queue if preempted
#SBATCH --nodes=1
#SBATCH --job-name=q1_fe01              # Assign an short name to your job
#SBATCH --cpus-per-task=4          # Cores per task (>1 if multithread tasks)
#SBATCH --mem=16GB                 # Real memory (RAM) required
#SBATCH --gres=gpu:4               # Generic resources
#SBATCH --time=05:00:00            # Total run time limit (HH:MM:SS)
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=cg4177@nyu.edu # send me mail

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate dls

python /home/cg4177/t5-tuning/t5_gsm8k.py --batch_size 4 --epochs 2 --model_name t5-base