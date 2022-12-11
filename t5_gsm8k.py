import os
import pytorch_lightning as pl
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
import json
import logging
import random
from itertools import chain
from string import punctuation
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from gsm8k_dataload import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='type of t5 model')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--epochs', type=int, help='number of epochs used in training')

args = parser.parse_args()


TRAIN_DATA_JSON = "grade-school-math/grade_school_math/data/train.jsonl"
TEST_DATA_JSON = "grade-school-math"
# using the base T5 model having 222M params
# MODEL_NAME ='t5-base'
MODEL_NAME = args['model_name']
BATCH_SIZE = args['batch_size']
# BATCH_SIZE = 4
EPOCHS = args['epochs']

# EPOCHS = 3


df = extract_questions_and_answers(path=TRAIN_DATA_JSON)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

train_df, val_df = train_test_split(df, test_size=0.05)

data_module = GSMDataModule(train_df, val_df, tokenizer, batch_size=BATCH_SIZE)
data_module.setup()

"""Fine Tuning t5"""

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict = True)

model = GSMQAModel()

# To record the best performing model using checkpoint

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

#logger = TensorBoardLogger("training-logs", name="bio-qa")
trainer = pl.Trainer(
    #logger = logger,
    checkpoint_callback=checkpoint_callback,
    max_epochs=EPOCHS,
    gpus=1,
    progress_bar_refresh_rate = 30
)

trainer.fit(model, data_module)

trainer.test()  # evaluate the model according to the last checkpoint

trained_model = GSMQAModel.load_from_checkpoint("checkpoints/best-checkpoint.ckpt")
trained_model.freeze() #
