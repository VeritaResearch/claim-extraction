from multiprocessing.spawn import prepare
import os
from os.path import join, dirname, pardir

import sys

PROJ_PATH = join(dirname(__file__), pardir)
sys.path.insert(0, PROJ_PATH)

from src.utils import load_model

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    ModernBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
import pandas as pd
from datasets import Dataset
import numpy as np
from sklearn.metrics import f1_score

cuda = False
cache_dir = "../assets/pretrained-models"
model_path = "answerdotai/ModernBERT-base"
output_dir = "../assets/finetuned-models/ModernBERT-claim-detection"

id2label = {0: "No", 1: "Yes"}
label2id = {"No": 0, "Yes": 1}

model = ModernBertForSequenceClassification.from_pretrained(
    model_path, cache_dir=cache_dir, id2label=id2label, label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

if cuda:
    model.to("cuda")

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt")

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=2,
    bf16=True, # bfloat16 training 
    optim="adamw_torch",
    # logging & evaluation strategies
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    #use_mps_device=True,
    metric_for_best_model="f1",
)

def prepare_dataset():
    dataset = pd.read_json('../data/Claimbuster/train.json')
    dataset = dataset.filter(items=['text','label'])
    dataset = dataset.rename(columns={"label": "labels"})
    
    dataset = dataset.sample(n=10,random_state=42)

    train_dataset = Dataset.from_pandas(dataset)
    tokenized_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])

    return tokenized_dataset

if __name__ == "__main__":
    print("DEBUG::preparing dataset")
    train_dataset = prepare_dataset()

    print("DEBUG::instantiating trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    print("DEBUG::starting training")
    trainer.train()
    print("DEBUG::saving model")
    trainer.save_model()