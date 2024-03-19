# -*- coding: utf-8 -*-
"""
This script is used to 'translate' MSA sentences opus100 to dialectal Arabic using the AraT5v2-MSA-Dialect model.
The translated sentences are then saved to a json file and pushed to the Hugging Face Hub.
"""
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
)


HF_TOKEN = ""


def remove_parenthesis(example):
    if "(" in example["msa"] and ")" in example["msa"] and "(" in example["dialect"] and ")" in example["dialect"]:
        return example
    example["msa"] = example["msa"].replace("(", "").replace(")", "")
    return example


model_name = "Murhaf/AraT5v2-MSA-Dialect"
tokenizer = T5Tokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN)

dataset = load_dataset("opus100", "ar-en", split="train")

dataset = dataset.filter(lambda example: 5 < len(example["translation"]["ar"]) < 450)
dataset = dataset.filter(lambda x: "{" not in x["translation"]["ar"])

ds = DataLoader(dataset, batch_size=256, shuffle=False)

# Check if GPU is available, and move model and tokenizer to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prefix = "ترجمة للعامية: "
dialect = []

# Batch size for processing
batch_size = 64

# Iterate over the sentences in batches
for batch in tqdm(ds):
    batch_sentences = batch["translation"]['ar']

    # Tokenize batch of sentences
    inputs = tokenizer(
        [prefix + sent for sent in batch_sentences],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Move inputs to appropriate device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=60)

    # Decode and store outputs
    decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    dialect.extend(decoded_outputs)

data = dataset.to_dict()

# Update dataframe once at the end
df = pd.DataFrame(data["translation"])
df["dialect"] = dialect

df.to_json(
    "dialectal_opus.json",
    orient="records",
    lines=True,
    force_ascii=False,
)

dataset = load_dataset("json", data_files="dialectal_opus.json")
dataset = dataset.map(remove_parenthesis)
dataset.push_to_hub("opus100_msa_dialect_silver", private=True, token=HF_TOKEN)
