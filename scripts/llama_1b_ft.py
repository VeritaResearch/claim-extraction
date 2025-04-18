import os
from os.path import join, dirname, pardir

import sys

PROJ_PATH = join(dirname(__file__), pardir)
sys.path.insert(0, PROJ_PATH)

import pandas as pd
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompts import FactcheckGPT_SYSTEM_PROPMT, CHECKWORTHY_PROMPT, SPECIFY_CHECKWORTHY_CATEGORY_PROMPT

cuda = True
cache_dir = "../assets/pretrained-models"
model_path = "meta-llama/Llama-3.2-1B-Instruct"
output_dir="../assets/finetuned-models/Llama-3.2-1B-Instruct-SFT"
full_data_path = '../data/ours/train.csv'

def prepare_dataset():
    dataset = pd.read_csv(full_data_path)
    dataset = dataset.dropna()

    dataset["label_yes_no"] = dataset["label"].apply(
        lambda x: "Yes" if x == 1 else "No"
    )
    dataset["messages"] = dataset.apply(
        lambda row: [
            {"role": "system", "content": FactcheckGPT_SYSTEM_PROPMT},
            {"role": "user", "content": CHECKWORTHY_PROMPT.format(texts = ('["' + row['text'] + '"]'))}
        ],
        axis=1,
    )
    dataset = dataset.filter(items=["messages"])
    train_dataset = Dataset.from_pandas(dataset)
    return train_dataset

training_args = SFTConfig(
    output_dir=output_dir,
    logging_steps=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=3,
    bf16=True,
    ) # bfloat16 training 

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=cache_dir, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=cache_dir, use_safetensors=True
    )

    if cuda:
        model.to("cuda")

    train_dataset = prepare_dataset()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config)
    trainer.train()
    trainer.save_model()