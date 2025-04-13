import os
from os.path import join, dirname, pardir

import sys

PROJ_PATH = join(dirname(__file__), pardir)
sys.path.insert(0, PROJ_PATH)

from src.utils import load_model, prepare_eval_dataset
from src.eval import eval

cache_dir = "../assets/pretrained-models"
model_path = "meta-llama/Llama-3.2-1B-Instruct"

if __name__ == "__main__":
    model, tokenizer = load_model(
        model_path=model_path, cache_dir=cache_dir, cuda=False
    )

    eval_dataset = prepare_eval_dataset(sample=10)

    results_df = eval(model, tokenizer, eval_dataset, verbose=True)
