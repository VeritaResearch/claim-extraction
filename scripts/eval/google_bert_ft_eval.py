from functools import cache
import os
from os.path import join, dirname, pardir
from pickle import TRUE

import sys

PROJ_PATH = join(dirname(__file__), pardir)
sys.path.insert(0, PROJ_PATH)

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, pipeline
import torch

verbose = True
cuda = False
full_data_path = "../../data/ours/test.csv"
cache_dir = "../../assets/finetuned-models"
full_save_path = "../../results/google_bert_ft_eval.csv"
 
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir=cache_dir)
    
    classifier = pipeline(
        task="text-classification", 
        model="../assets/finetuned-models/bert-base-uncased-claim-detection",
        tokenizer=tokenizer,
    )
    
    if cuda:
        classifier.to("cuda")

    eval_dataset = pd.read_csv(full_data_path)
    eval_dataset = eval_dataset.to_dict(orient="records")

    results_df = pd.DataFrame(columns=["text", "label", "pred_str", "pred"])

    for i, eval_instance in enumerate(tqdm(eval_dataset)):

        pred_str = classifier(eval_instance['text'])[0]['label']

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
            results_df.to_csv(full_save_path,index=None)

    if verbose:
        y_true = results_df['label']
        y_pred = results_df['pred']

        print("DEBUG::results::eval f1", f1_score(y_true,y_pred))
        print("DEBUG::resutls::pred value counts", results_df["pred"].value_counts())

