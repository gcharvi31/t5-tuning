# argparse makes it easier to write user friendly command line interfaces
import argparse
# package for faster file name matching
import glob
# makiing directories for data 
import os
# reading json files as the data is present in json files
import json
# time module for calculating the model runtime
import time
# Allows writing status messages to a file
import logging
# generate random float numbers uniformly
import random
# regex module for text 
import re
# module provides various functions which work on 
# iterators too produce complex iterators
from itertools import chain
from string import punctuation

# pandas for data manipulation
import pandas as pd
# numpy for array operations
import numpy as np
# PyTorch
import torch
# provides various classes representing file system paths
# with appropriate semantics
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# splitting the data 
from sklearn.model_selection import train_test_split
# ANSII color formatting for ouput in terminal
from termcolor import colored
# wrapping paragraphs into string
import textwrap

# model checkpoints in pretrained model
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

import subprocess
from pytorch_lightning.callbacks import TQDMProgressBar
import pickle

# Seeds all the processes including numpy torch and other imported modules - makes for better comparisions
pl.seed_everything(0)


# Configuration
num_of_gpus = 2
fp_precision = 32
num_of_workers = 2
logger_fname = 'log_compute_model_5.csv'
model_name = "model_5"
MODEL_NAME ='t5-base'
BATCH_SIZE = 4
N_EPOCHS = 20
val_losses = []

logger_pid = subprocess.Popen(
    ['python', '../log_gpu_cpu_stats.py',
     logger_fname,
     '--loop',  '30',  # Interval between measurements, in seconds (optional, default=1)
    ])
print('Started logging compute utilisation')

def extract_questions_and_answers(factoid_path = Path):
  with factoid_path.open() as json_file:
    data = json.load(json_file)
    questions = data['data'][0]['paragraphs']
    data_rows = []
    for question in questions:
      context = question['context']
      for question_and_answers in question['qas']:
        question = question_and_answers['question']
        answers = question_and_answers['answers']
        for answer in answers:
          answer_text = answer['text']
          answer_start = answer['answer_start']
          answer_end = answer['answer_start'] + len(answer_text)  #Gets the end index of each answer in the paragraph
          
          data_rows.append({
                "question" : question,
                "context"  : context,
                "answer_text" : answer_text,
                "answer_start" : answer_start,
                "answer_end" : answer_end
            })
    return pd.DataFrame(data_rows)


factoid_paths = sorted(list(Path('../BioASQ/').glob('BioASQ-train-*')))
dfs = []

for factoid_path in factoid_paths:
  df = extract_questions_and_answers(factoid_path)
  dfs.append(df)

df = pd.concat(dfs)

# Dropping all the rows with repeated context and questions pairs.

df = df.drop_duplicates(subset=["context"]).reset_index(drop=True)

# using the base T5 model having 222M params
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

class BioQADataset(Dataset):
  def __init__(
      self,
      data:pd.DataFrame,
      tokenizer:T5Tokenizer,
      source_max_token_len: int = 396,
      target_max_token_len: int = 32,

      ):
    
    self.data =  data
    self.tokenizer =  tokenizer
    self.source_max_token_len =  source_max_token_len
    self.target_max_token_len =  target_max_token_len


  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    source_encoding = tokenizer(
      data_row['question'],
      data_row['context'],
      max_length=self.source_max_token_len,
      padding='max_length',
      truncation="only_second",
      return_attention_mask=True,
      add_special_tokens=True,
      return_tensors="pt"
      )
    
    target_encoding = tokenizer(
      data_row['answer_text'],
      max_length=self.target_max_token_len,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      add_special_tokens=True,
      return_tensors="pt"
      )
    
    labels = target_encoding['input_ids']
    labels[labels==0] = -100

    return dict(
        question=data_row['question'],
        context=data_row['context'],
        answer_text=data_row['answer_text'],
        input_ids=source_encoding["input_ids"].flatten(),
        attention_mask=source_encoding['attention_mask'].flatten(),
        labels=labels.flatten()
    )

class BioDataModule(pl.LightningDataModule):
  def __init__(
      self,
      train_df: pd.DataFrame,
      test_df: pd.DataFrame,
      tokenizer:T5Tokenizer,
      batch_size: int = 8,
      source_max_token_len: int = 396,
      target_max_token_len: int = 32,
      ):
    super().__init__()
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.batch_size = batch_size
    self.source_max_token_len = source_max_token_len
    self.target_max_token_len = target_max_token_len

  def setup(self, stage=None):
    self.train_dataset = BioQADataset(
        self.train_df,
        self.tokenizer,
        self.source_max_token_len,
        self.target_max_token_len
        )

    self.test_dataset = BioQADataset(
    self.test_df,
    self.tokenizer,
    self.source_max_token_len,
    self.target_max_token_len
    )
 
  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=num_of_workers
        )
  def val_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        num_workers=num_of_workers
        )

  def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=1,
        num_workers=num_of_workers
        )


train_df, val_df = train_test_split(df, test_size=0.05)
data_module = BioDataModule(train_df, val_df, tokenizer, batch_size=BATCH_SIZE)
data_module.setup()

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict = True)


class BioQAModel(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)


  def forward(self, input_ids, attention_mask, labels=None):
    output = self.model(
        input_ids, 
        attention_mask=attention_mask,
        labels=labels)

    return output.loss, output.logits

  def training_step(self, batch, batch_idx, dataloader_idx=0):
    input_ids = batch['input_ids']
    attention_mask=batch['attention_mask']
    labels = batch['labels']
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions":outputs, "labels": labels}

  def validation_step(self, batch, batch_idx, dataloader_idx=0):
    input_ids = batch['input_ids']
    attention_mask=batch['attention_mask']
    labels = batch['labels']
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    val_losses.append(loss.item())
    return loss

  def test_step(self, batch, batch_idx, dataloader_idx=0):
    input_ids = batch['input_ids']
    attention_mask=batch['attention_mask']
    labels = batch['labels']
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):

    optimizer = AdamW(self.parameters(), lr=0.0001)
    return optimizer

model = BioQAModel() 

# To record the best performing model using checkpoint

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints_2",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)


trainer = pl.Trainer(
    callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=30)],
    accelerator="gpu",
    max_epochs=N_EPOCHS,
    devices=num_of_gpus,
    strategy="ddp_sharded",
    precision=fp_precision
)


print("Starting training")
training_start = time.time()
trainer.fit(model, data_module)
training_time = time.time()-training_start
print("Training done")

trained_model = BioQAModel.load_from_checkpoint("checkpoints_2/best-checkpoint.ckpt")
trained_model.freeze()

def generate_answer(question):
  source_encoding=tokenizer(
      question["question"],
      question['context'],
      max_length = 396,
      padding="max_length",
      truncation="only_second",
      return_attention_mask=True,
      add_special_tokens=True,
      return_tensors="pt"

  )

  generated_ids = trained_model.model.generate(
      input_ids=source_encoding["input_ids"],
      attention_mask=source_encoding["attention_mask"],
      num_beams=1,  # greedy search
      max_length=80,
      repetition_penalty=2.5,
      early_stopping=True,
      use_cache=True)
  
  preds = [
          tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
          for generated_id in generated_ids
  ]

  return "".join(preds)

sample_question = val_df.iloc[12]
pred_ans = generate_answer(sample_question)  # Predicted answer

print("Question: ", sample_question["question"])
print("Ans: ", pred_ans)

logger_pid.terminate()
print('Terminated the compute utilisation logger background process')

results = {"val_losses": val_losses, "training_time": training_time}

with open(model_name+".pickle", 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
