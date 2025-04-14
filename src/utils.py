from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from datasets import Dataset


def load_model(
    model_path: str,
    cache_dir: str = None,
    token: str = None,
    cuda: bool = False,
):
    """
    Load a model from Hugging Face model hub.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path, token=token, cache_dir=cache_dir, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, token=token, cache_dir=cache_dir, use_safetensors=True
    )

    if cuda:
        model.to("cuda")

    return model, tokenizer


def prepare_sft_train_dataset(
    data_path: str = "../data/Claimbuster/train.json", sample=None
):
    """
    Perpare the dataset for the SFTTrainer
    Based on https://huggingface.co/docs/trl/main/en/sft_trainer#dataset-format-support
    """
    dataset = pd.read_json(data_path)
    dataset["label_yes_no"] = dataset["label"].apply(
        lambda x: "Yes" if x == 1 else "No"
    )
    dataset["messages"] = dataset.apply(
        lambda row: [
            {
                "role": "user",
                "content": f"### Sentence: {row['text']}\n### Instruction: Is the sentence a factual claim that could be checked by a fact-cheker?",
            },
            {'role': 'system', 'content': 'Only respond Yes or No'},
            {"role": "assistant", "content": f"{row['label_yes_no']}"},
        ],
        axis=1,
    )
    dataset = dataset.filter(items=["messages"])
    if sample:
        dataset = dataset.sample(n=sample, random_state=42)

    train_dataset = Dataset.from_pandas(dataset)

    return train_dataset


def prepare_eval_dataset(data_path: str = "../data/Claimbuster/test.json", sample=None):
    """
    Loads and processes a dataset for evaluation of factual claim classification.

    Args:
        data_path (str): Path to the JSON file containing the dataset.
                            Defaults to "../data/Claimbuster/test.json".
        sample (int, optional): If provided, randomly samples this number of rows
                                from the dataset for evaluation. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - 'prompt': A list of message dictionaries formatted for model input,
                        simulating a user asking if a sentence is a factual claim.
            - 'completion': A string ("Yes" or "No") representing the expected response
                            based on the original label.
    """

    dataset = pd.read_json(data_path)
    dataset["prompt"] = dataset["text"].apply(
        lambda x: [
            {
                "role": "user",
                "content": f"### Sentence: {x}\n### Instruction: Is the sentence a factual claim that could be checked by a fact-cheker?",
            },
            {'role': 'system', 'content': 'Only respond Yes or No'},
            {"role": "assistant", "content": ""},
        ]
    )
    dataset["completion"] = dataset["label"].apply(lambda x: "Yes" if x == 1 else "No")
    dataset = dataset.filter(items=["prompt", "completion"])

    if sample:
        dataset = dataset.sample(n=sample, random_state=42)

    return dataset