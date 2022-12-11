import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

'''
optimizer - AdamW
T5 Conditional Generator in which we'll give conditions
T5 tokenizer because it is fast training the model without a learning rate
'''
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
SOURCE_MAX_TOKEN_LEN = 500
TARGET_MAX_TOKEN_LEN = 20


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def extract_questions_and_answers(path):
  data = read_jsonl(path=path)
  questions = [i['question'] for i in data]
  answer_text = ["The answer is "+ ANS_RE.search(i['answer']).group(1).strip().replace(",", "") for i in data]

  assert(len(questions) == len(answer_text))

  data_rows = []
  for i in range(len(questions)):
    data_rows.append({
        "question": questions[i],
        "answer_text": answer_text[i]
        })
  return pd.DataFrame(data_rows)


class GSM8kQADataset(Dataset):
  def __init__(
      self,
      data:pd.DataFrame,
      tokenizer:T5Tokenizer,
      source_max_token_len: int = SOURCE_MAX_TOKEN_LEN,
      target_max_token_len: int = TARGET_MAX_TOKEN_LEN,
      ):
    
    self.data =  data
    self.tokenizer =  tokenizer
    self.source_max_token_len =  source_max_token_len
    self.target_max_token_len =  target_max_token_len


  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    source_encoding =self.tokenizer(
      data_row['question'],
      max_length=self.source_max_token_len,
      padding='max_length',
      truncation="only_second",
      return_attention_mask=True,
      add_special_tokens=True,
      return_tensors="pt"
      )
    
    target_encoding = self.tokenizer(
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
        answer_text=data_row['answer_text'],
        input_ids=source_encoding["input_ids"].flatten(),
        attention_mask=source_encoding['attention_mask'].flatten(),
        labels=labels.flatten()
    )


class GSMDataModule(pl.LightningDataModule):
  def __init__(
      self,
      train_df: pd.DataFrame,
      test_df: pd.DataFrame,
      tokenizer:T5Tokenizer,
      batch_size: int = 8,
      source_max_token_len: int = SOURCE_MAX_TOKEN_LEN,
      target_max_token_len: int = TARGET_MAX_TOKEN_LEN,
      ):
    super().__init__()
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.batch_size = batch_size
    self.source_max_token_len = source_max_token_len
    self.target_max_token_len = target_max_token_len

  def setup(self):
    self.train_dataset = GSM8kQADataset(
        self.train_df,
        self.tokenizer,
        self.source_max_token_len,
        self.target_max_token_len
        )

    self.test_dataset = GSM8kQADataset(
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
        num_workers=4
        )
  def val_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        num_workers=4
        )

  def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=1,
        num_workers=4
        )


class GSMQAModel(pl.LightningModule):
  def __init__(self, model_name:str):
    super().__init__()
    self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)


  def forward(self, input_ids, attention_mask, labels=None):
    output = self.model(
        input_ids, 
        attention_mask=attention_mask,
        labels=labels)

    return output.loss, output.logits

  def training_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask=batch['attention_mask']
    labels = batch['labels']
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions":outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask=batch['attention_mask']
    labels = batch['labels']
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask=batch['attention_mask']
    labels = batch['labels']
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=0.0001)
    return optimizer


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