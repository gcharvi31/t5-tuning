import numpy as np
import pandas as pd
import json

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
