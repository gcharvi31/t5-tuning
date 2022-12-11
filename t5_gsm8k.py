import os
import time
import argparse
import logging
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from transformers import T5Tokenizer
from gsm8k_dataload import extract_questions_and_answers, GSMDataModule, GSMQAModel, generate_answer
from params import meta_params

# Seeds all the processes including numpy torch and other imported modules - makes for better comparisions
pl.seed_everything(0)

### Download gsm8k from Github into scratch folder
RAW_DATA_DIR = meta_params["RAW_DATA_DIR"]
Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
MODEL_CHKPT_DIR = meta_params["CHKPT_DIR"]
Path(MODEL_CHKPT_DIR).mkdir(parents=True, exist_ok=True)
RUN_LOGS_DIR = meta_params["RUN_LOGS_DIR"]
Path(RUN_LOGS_DIR).mkdir(parents=True, exist_ok=True)

# Create and configure logger
filename = datetime.now().strftime('gsm8k_%H_%M_%d_%m_%Y')
logging.basicConfig(filename = f'{RUN_LOGS_DIR}/{filename}.log',
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='type of t5 model', default="t5-base")
parser.add_argument('--batch_size', type=int, help='batch size', default=4)
parser.add_argument('--epochs', type=int, help='number of epochs used in training', default=3)
parser.add_argument('--fp_precision', type=int, help='floating point precision', default=16)

args = parser.parse_args()

TRAIN_DATA_JSON = f"{RAW_DATA_DIR}/grade-school-math/grade_school_math/data/train.jsonl"
TEST_DATA_JSON = f"{RAW_DATA_DIR}/grade-school-math/grade_school_math/data/test.jsonl"

MODEL_NAME = args.model_name
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
FP_PRECISION = args.fp_precision

logging.info(f"Using {MODEL_NAME} as pretrained base")

logging.debug("Generating tokenizer from pretrained model")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

logging.debug("Loading raw data from json to dataframe")
df = extract_questions_and_answers(path=TRAIN_DATA_JSON)
train_df, val_df = train_test_split(df, test_size=0.05)
test_df = extract_questions_and_answers(path=TEST_DATA_JSON)

logging.debug(f"Train data size: {train_df.shape}, Val data size: {val_df.shape}, Test data size: {test_df.shape}")

logging.debug("Generating train and val dataset objects")
data_module = GSMDataModule(train_df, val_df, test_df, tokenizer, batch_size=BATCH_SIZE)
data_module.setup()

logging.debug(f"Loading {MODEL_NAME} pretrained model")
model = GSMQAModel(MODEL_NAME=MODEL_NAME)

# To record the best performing model using checkpoint

CHKPT_FILENAME = f"gsm8k_{MODEL_NAME}"
checkpoint_callback = ModelCheckpoint(
    dirpath=MODEL_CHKPT_DIR,
    filename=CHKPT_FILENAME,
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

trainer = pl.Trainer(
    callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=30)],
    max_epochs=EPOCHS,
    gpus=1,
    precision=FP_PRECISION
)

trainer.fit(model, data_module)

# trainer.test()  # evaluate the model according to the last checkpoint

trained_model = GSMQAModel(MODEL_NAME=MODEL_NAME).load_from_checkpoint(f"{MODEL_CHKPT_DIR}/{CHKPT_FILENAME}.ckpt")
trained_model.freeze()

val_losses = trained_model.val_losses

sample_question = val_df.iloc[12]
pred_ans = generate_answer(sample_question, tokenizer=tokenizer, trained_model=trained_model)  # Predicted answer

print("Question: ", sample_question["question"])
print("Ans: ", pred_ans)

results = {
    "model_name": MODEL_NAME,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "fp_precision": FP_PRECISION,
    "val_losses": val_losses}

results_filename = f'{RUN_LOGS_DIR}/{filename}.pkl'
with open (results_filename, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
