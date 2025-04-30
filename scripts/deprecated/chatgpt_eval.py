import os
from os.path import join, dirname, pardir

import sys

PROJ_PATH = join(dirname(__file__), pardir)
sys.path.insert(0, PROJ_PATH)

import yaml
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

# from src.utils import prepare_chat_gpt_eval_dataset
# from src.eval import eval

# cache_dir = "../assets/pretrained-models"
# model_path = "meta-llama/Llama-3.2-1B-Instruct"

verbose = True

with open("creds.yaml") as f:
    creds = yaml.load(f, Loader=yaml.FullLoader)


def prepare_chat_gpt_eval_dataset(
    data_path: str = "../data/Claimbuster/test.json", sample=None
):
    """
    Perpare the dataset for the openai API
    """
    dataset = pd.read_json(data_path)
    dataset["completion"] = dataset["label"].apply(lambda x: "Yes" if x == 1 else "No")    
    dataset["messages"] = dataset.apply(
        lambda row: [
            {
                "role": "user",
                "content": f"### Sentence: {row['text']}\n### Instruction: Is the sentence a factual claim that could be checked by a fact-cheker? Only respond Yes or No",
            },
            #{"role": "system", "content": "Only respond Yes or No"},
        ],
        axis=1,
    )
    dataset = dataset.filter(items=["messages","completion"])
    if sample:
        dataset = dataset.sample(n=sample, random_state=42)

    return dataset


if __name__ == "__main__":

    eval_dataset = prepare_chat_gpt_eval_dataset()
    eval_dataset = eval_dataset.to_dict(orient="records")

    client = OpenAI(api_key=creds["token"], organization=creds["org_key"])

    results_df = pd.DataFrame(columns=["prompt", "completion", "pred", "eval"])

    for i, eval_instance in enumerate(tqdm(eval_dataset)):
        
        response = client.chat.completions.create(
            messages=eval_instance['messages'],
            model="o1-mini",
            #temperature=0
            #max_completion_tokens=3
        )

        pred = response.choices[0].message.content
        
        if verbose:
            print("DEBUG::prompt::", eval_instance["messages"])
            print("DEBUG::completion::", eval_instance["completion"])
            print("DEBUG::pred::", pred)

        r = {
            "prompt": eval_instance["messages"],
            "completion": eval_instance["completion"],
            "pred": pred,
            "eval": 1 if (pred in eval_instance["completion"]) else 0,
        }
        results_df.loc[len(results_df)] = r

    if verbose:
        print("DEBUG::results::eval mean", results_df["eval"].mean())
        print("DEBUG::resutls::pred value counts", results_df["pred"].value_counts())

    results_df.to_csv('../results/gpt-o1-mini-eval.csv',index=None)