module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate idls_project_1

WORLD_SIZE=4 python -m torchrun /home/cg4177/t5-tuning/t5_gsm8k.py \
    --identifier model_1
    --model_name t5-base \
    --batch_size 4 \
    --epochs 5 \
    --fp_precision 32 \
    --devices 2 \
