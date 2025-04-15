import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

def eval(model, tokenizer, eval_dataset, cuda=False, verbose=False):
    """
    Evaluate a model with an eval_dataset
    Eval_dataset should have two columns: prompt and completion
    Prompt should be in messages format
    """
    eval_dataset = eval_dataset.to_dict(orient="records")

    results_df = pd.DataFrame(columns=["prompt", "completion", "pred", "eval"])
    for i, eval_instance in enumerate(tqdm(eval_dataset)):
        chat_template_input_ids = tokenizer.apply_chat_template(
            eval_instance["prompt"],
            tokenize=True,
            continue_final_message=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        # The following reformatting line is necessary because continue_final_message doesn't work with empty messages
        chat_template_input_ids = chat_template_input_ids[0, :-1].reshape(1, -1)

        if cuda:
            chat_template_input_ids = chat_template_input_ids.cuda()

        with torch.no_grad():
            logits = model(chat_template_input_ids, use_cache=True)["logits"]

        del chat_template_input_ids

        next_token = torch.argmax(logits[0, -1]).item()
        next_token_text = tokenizer.convert_ids_to_tokens(next_token)

        if verbose:
            print(f"DEBUG::i::{i} / {len(eval_dataset)}")
            print("DEBUG::prompt::", eval_instance["prompt"])
            print("DEBUG::completion::", eval_instance["completion"])
            print("DEBUG::pred::", next_token_text)

        r = {
            "prompt": eval_instance["prompt"],
            "completion": eval_instance["completion"],
            "pred": next_token_text,
            "eval": 1 if (next_token_text in eval_instance["completion"]) else 0,
        }
        results_df.loc[len(results_df)] = r

    if verbose:
        print("DEBUG::results::eval mean", results_df["eval"].mean())
        print("DEBUG::resutls::pred value counts", results_df["pred"].value_counts())

    return results_df


def ModernBERT_eval(full_model_path, tokenizer, data_path: str = "../data/Claimbuster/test.json", cuda=False, verbose=False):
    """
    Evaluate a model with an eval_dataset
    Eval_dataset should have two columns: prompt and completion
    Prompt should be in messages format
    """
    dataset = pd.read_json(data_path)
    dataset['label_text'] = dataset['label'].apply(lambda x: 'Yes' if x == 1 else 'No')
    eval_dataset = dataset.to_dict(orient="records")

    if cuda:
        device = 'cuda:1'
    else:
        device = 0

    classifier = pipeline(
        task="text-classification", 
        model=full_model_path,
        tokenizer=tokenizer,
        device=device
    )

    results_df = pd.DataFrame(columns=["claim", "label", "pred", "eval"])
    for i, eval_instance in enumerate(tqdm(eval_dataset)):
        pred = classifier(eval_instance['text'])[0]['label']

        if verbose:
            print(f"DEBUG::i::{i} / {len(eval_dataset)}")
            print("DEBUG::claim::", eval_instance["text"])
            print("DEBUG::label::", eval_instance["label_text"])
            print("DEBUG::pred::", pred)

        r = {
            "claim": eval_instance["text"],
            "label": eval_instance["label_text"],
            "pred": pred,
            "eval": 1 if (pred in eval_instance['label_text']) else 0,
        }
        results_df.loc[len(results_df)] = r

    if verbose:
        print("DEBUG::results::eval mean", results_df["eval"].mean())
        print("DEBUG::resutls::pred value counts", results_df["pred"].value_counts())

    return results_df
