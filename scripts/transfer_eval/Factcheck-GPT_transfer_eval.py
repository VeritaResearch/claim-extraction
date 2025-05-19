import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import yaml
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score

from src.prompts import FactcheckGPT_SYSTEM_PROPMT, CHECKWORTHY_PROMPT, SPECIFY_CHECKWORTHY_CATEGORY_PROMPT
from src.data_utils import process_CheckThat

verbose = True
full_save_path = '../../results/Factcheck-GPT_transfer_eval.csv'

with open("creds.yaml") as f:
    creds = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == "__main__":
    client = OpenAI(api_key=creds["token"], organization=creds["org_key"])

    eval_dataset = process_CheckThat()
    eval_dataset = eval_dataset.to_dict(orient="records")

    results_df = pd.DataFrame(columns=["text", "label", "pred_str", "pred"])

    for i, eval_instance in enumerate(tqdm(eval_dataset)):

        texts_formatted = '["' + eval_instance['text'] + '"]'
        messages = [
            {"role": "system", "content": FactcheckGPT_SYSTEM_PROPMT},
            {"role": "user", "content": CHECKWORTHY_PROMPT.format(texts = texts_formatted)}
        ]

        response = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            temperature=0,
            max_completion_tokens=10,
        )

        pred_str = response.choices[0].message.content

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

    y_true = results_df['label']
    y_pred = results_df['pred']

    if verbose:
        print("DEBUG::results::eval f1", f1_score(y_true,y_pred))
        print("DEBUG::resutls::pred value counts", results_df["pred"].value_counts())

