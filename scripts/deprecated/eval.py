import os
from os.path import join, dirname, pardir

import sys

PROJ_PATH = join(dirname(__file__), pardir)
sys.path.insert(0, PROJ_PATH)

from src.utils import load_model, prepare_eval_dataset
from src.eval import eval, ModernBERT_eval
from transformers import AutoTokenizer

cache_dir = "../assets/pretrained-models"
model_path = "../assets/finetuned-models/ModernBERT-claim-detection"

if __name__ == "__main__":

    if 'ModernBERT' in model_path:
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", cache_dir=cache_dir)
        results_df = ModernBERT_eval(model_path, tokenizer,verbose=True)
    else:
        model, tokenizer = load_model(model_path=model_path, cache_dir=cache_dir, cuda=False)

        eval_dataset = prepare_eval_dataset(sample=10)
        results_df = eval(model, tokenizer, eval_dataset, cuda=False, verbose=True)
