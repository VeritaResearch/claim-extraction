import os
from os.path import join, dirname, pardir

import sys

PROJ_PATH = join(dirname(__file__), pardir)
sys.path.insert(0, PROJ_PATH)

from trl import SFTConfig, SFTTrainer
from src.utils import load_model, prepare_sft_train_dataset

cache_dir = "../assets/pretrained-models"
model_path = "meta-llama/Llama-3.2-1B-Instruct"
output_dir="../assets/finetuned-models/Llama-3.2-1B-Instruct-SFT"

if __name__ == "__main__":
    training_args = SFTConfig(output_dir=output_dir, logging_steps=10)

    model, tokenizer = load_model(
        model_path=model_path, cache_dir=cache_dir, cuda=False
    )
    train_dataset = prepare_sft_train_dataset(sample=3)

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    trainer.save_model()