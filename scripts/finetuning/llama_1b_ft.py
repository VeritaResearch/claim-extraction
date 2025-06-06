import os
from os.path import join, dirname, pardir

import sys

PROJ_PATH = join(dirname(__file__), pardir)
sys.path.insert(0, PROJ_PATH)

import pandas as pd
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.prompts import LLAMA_SYSTEM_PROMPT, LLAMA_CHECKWORTHY_PROMPT

cuda = torch.cuda.is_available()
cache_dir = "../../assets/pretrained-models"
model_path = "meta-llama/Llama-3.2-1B-Instruct"
output_dir="../../assets/finetuned-models/Llama-3.2-1B-Instruct-SFT"
full_data_path = "../../data/ours/train.csv"

def prepare_dataset():
    dataset = pd.read_csv(full_data_path)
    dataset = dataset.dropna()

    dataset["label_yes_no"] = dataset["label"].apply(
        lambda x: "Yes" if x == 1 else "No"
    )
    dataset["messages"] = dataset.apply(
        lambda row: [
            {"role": "system", "content": LLAMA_SYSTEM_PROMPT},
            {"role": "user", "content": LLAMA_CHECKWORTHY_PROMPT.format(texts = row['text'])},
            {"role": "assistant", "content": row['label_yes_no']}
        ],
        axis=1,
    )
        
    dataset = dataset.filter(items=["messages"])
    train_dataset = Dataset.from_pandas(dataset)
    return train_dataset

training_args = SFTConfig(
    output_dir=output_dir,
    logging_steps=100,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    torch_empty_cache_steps=4,
    learning_rate=5e-5,
    num_train_epochs=30,
    #fp16=True,
    #bf16=False,
    optim="adamw_torch",
    do_train=True,
    do_eval=False,
    gradient_checkpointing=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    #bias="none",
    task_type="CAUSAL_LM",
    target_modules = ["q_proj", "k_proj", "v_proj"]
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=cache_dir, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=cache_dir, use_safetensors=True, #quantization_config=bnb_config
    )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model = get_peft_model(model, lora_config)
    model.config.use_cache=False

    if cuda:
        model.to("cuda")
   
    total_parameters = 0
    for name, param in model.named_parameters():
        if 'lora' in name:
            # print(param.requires_grad)
            total_parameters += param.numel()

    print(f"Total lora parameters: {total_parameters}")

    #for name, param in model.named_parameters():
    #    if 'lora' not in name:
    #        print(f'Freezing non-LoRA parameter {name} | {param.requires_grad}')
    #        param.requires_grad = False

    train_dataset = prepare_dataset()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        )
    # Can check this here: https://github.com/unslothai/unsloth/issues/1802
    trainer.train()
    trainer.save_model()
