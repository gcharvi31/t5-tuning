# Evaluating different training methods on QA datasets

This project was done as a part of the Intro to Deep Learning Course taken by Prof. Parijat Dube at NYU Courant. We aim to compare and contrast a variety of training configurations on the task of fine tuning a T5 model on QA datasets (BioQA and GSM) to understand their performance and gain insight on when to use what methods.

## Files

`environment.yml` - the conda virtual environment (contains all the package requirements) 

`gsm8k_dataload.py` - python script to download the GSM dataset

`gsm_8k.py` - python script for fine-tuning the T5 model on the GSM dataset

`bio_download.sh` - bash script to download the BioQA dataset

`bio_train_1.py` - python script for fine-tuning the T5 model on the BioQA dataset on 1 GPU

`bio_train_2.py` - python script for fine-tuning the T5 model on the BioQA dataset on 2 GPUs

`log_gpu_cpu_stats.py` - helper script to log GPU usage stats

`/scripts` - SBATCH scripts used to run jobs on the HPC

`visualizations.ipynb` - notebook used to create graphs from the obtained results

## Running Instructions

1) Set up a virtual environment using the `environment.yml` file
2) Run the scripts required for downloading the dataset
3) Run the training scripts 
4) Use the `visualizations.ipynb` to compare the recorded metrics


## Configurations tested

| Model Number | Number of workers | Number of GPUs | FP Precision | Strategy       |
|--------------|-------------------|----------------|--------------|----------------|
| 1            | 1                 | 1              | 32           | -              |
| 2            | 2                 | 1              | 32           | -              |
| 3            | 2                 | 1              | 16           | -              |
| 4            | 2                 | 2              | 32           | Data Parallel  |
| 5            | 2                 | 2              | 32           | Model Parallel |
| 6 (t5-large) | 2                 | 2              | 32           | Model Parallel |


## Results



## Contributors

1) Charvi Gupta (gcharvi31)
2) Rushabh Musthyala (Rushabh10)

## References
1) https://medium.com/analytics-vidhya/t5-a-detailed-explanation-a0ac9bc53e51
2) https://medium.com/pytorch/pytorch-lightning-1-1-model-parallelism-training-and-more-logging-options-7d1e47db7b0b
3) https://devblog.pytorchlightning.ai/how-we-used-pytorch-lightning-to-make-our-deep-learning-pipeline-10x-faster-731bd7ad318a
4) https://analyticsindiamag.com/guide-to-question-answering-system-with-t5-transformer/