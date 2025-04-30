import os
from os.path import join, dirname, pardir
from pickle import TRUE

import sys

PROJ_PATH = join(dirname(__file__), pardir)
sys.path.insert(0, PROJ_PATH)

import yaml
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompts import LLAMA_SYSTEM_PROMPT, LLAMA_CHECKWORTHY_PROMPT

verbose = True
cuda = False
cache_dir = "../../assets/pretrained-models"
model_path = "meta-llama/Llama-3.2-1B-Instruct"
adapter_path= "../../assets/finetuned-models/Llama-3.2-1B-Instruct-SFT"
full_data_path = "../../data/ours/test.csv"
full_save_path= "../results/llama_1b_ft_eval.csv"

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        adapter_path, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=cache_dir, use_safetensors=True
    )

    if cuda:
        model.to("cuda")

    eval_dataset = pd.read_csv(full_data_path)
    eval_dataset = eval_dataset.to_dict(orient="records")

    results_df = pd.DataFrame(columns=["text", "label", "pred_str", "pred"])
    for i, eval_instance in enumerate(tqdm(eval_dataset)):

        messages = [
            {"role": "system", "content": LLAMA_SYSTEM_PROMPT},
            {"role": "user", "content": LLAMA_CHECKWORTHY_PROMPT.format(texts = eval_instance['text'])}
        ]

        chat_template_input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            continue_final_message=False,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        # The following reformatting line is necessary because continue_final_message doesn't work with empty messages
        #chat_template_input_ids = chat_template_input_ids[0, :-1].reshape(1, -1)

        if cuda:
            chat_template_input_ids = chat_template_input_ids.cuda()

        with torch.no_grad():
            pred_str = tokenizer.batch_decode(model.generate(chat_template_input_ids, max_new_tokens = 10))[0].split("<|start_header_id|>assistant<|end_header_id|>")[1]

        del chat_template_input_ids

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

