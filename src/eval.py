import pandas as pd
import torch
from tqdm import tqdm

def eval(model, tokenizer, eval_dataset, verbose=False):
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
        
        with torch.no_grad():
            logits = model(chat_template_input_ids, use_cache=True)["logits"]

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