#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --job-name=mod_four

module purge

module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate idls_project_1
cd /home/rm6416/project_scripts/new/

python train_2.py
