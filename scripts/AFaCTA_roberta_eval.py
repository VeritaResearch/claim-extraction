import os
from os.path import join, dirname, pardir
from pickle import TRUE

import sys

PROJ_PATH = join(dirname(__file__), pardir)
sys.path.insert(0, PROJ_PATH)

import yaml
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import Dataset

verbose = True
cuda = False
full_data_path = '../data/ours/test.csv'
cache_dir = "../assets/pretrained-models"
model_path = "JingweiNi/roberta-base-afacta"

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    
    if cuda:
        model.to("cuda")

    eval_dataset = pd.read_csv(full_data_path)
    eval_dataset = eval_dataset.to_dict(orient="records")

    results_df = pd.DataFrame(columns=["text", "label", "pred_str", "pred"])

    for i, eval_instance in enumerate(tqdm(eval_dataset)):

        inputs = tokenizer(eval_instance['text'], return_tensors="pt")

        with torch.no_grad():  # Disable gradient tracking for inference
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        probs = torch.nn.functional.softmax(logits, dim=1)

        model.config.id2label, model.config.label2id = {0: "No", 1: "Yes"}, {"No": 0, "Yes": 1}

        pred_str = model.config.id2label[predicted_class_id]

        if verbose:
            print("DEBUG::text::", eval_instance['text'])
            print("DEBUG::label::", eval_instance['label'])
            print("DEBUG::pred_str::", pred_str)

        r = {
            "text": eval_instance['text'],
            "label": eval_instance['label'],
            "pred_str": pred_str,
            "pred": 1 if ("Yes" in pred_str) else 0,
        }
        results_df.loc[len(results_df)] = r

        if i % 100 == 0:
            results_df.to_csv('../results/AFaCTA_roberta_eval.csv',index=None)

    y_true = results_df['label']
    y_pred = results_df['pred']

    if verbose:
        print("DEBUG::results::eval f1", f1_score(y_true,y_pred))
        print("DEBUG::resutls::pred value counts", results_df["pred"].value_counts())

