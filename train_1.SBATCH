#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch

module purge

module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate idls_project
cd /home/rm6416/project_scripts/

python train_1.py
