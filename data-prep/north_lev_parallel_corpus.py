"""
Assuming the North Levantine dataset is in the data folder, this script will convert it to a JSON file
and optionally upload it to the Hugging Face Hub.
"""

import pandas as pd
from datasets import load_dataset


UPLOAD_TO_HUB = True


def read_data(file_path):
    with open(file_path, "r") as file:
        data = [x.strip() for x in file.readlines()]
    return data


def prepare_dataset():
    data = {}
    data["msa"] = read_data("data/UFAL Parallel Corpus of North Levantine 1.0/ufal-nla-v1.arb")
    data["dialect"] = read_data("data/UFAL Parallel Corpus of North Levantine 1.0/ufal-nla-v1.apc")
    # dataframe from data
    df = pd.DataFrame(data)
    df.to_json("data/parallel_corpus_north_levantine.json", orient="records", lines=True)

    if UPLOAD_TO_HUB:
        dataset = load_dataset("json", data_files="data/parallel_corpus_north_levantine.json")
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        print(f"train shape {dataset['train'].shape}")
        print(f"val shape {dataset['test'].shape}")
        dataset_name = "parallel_corpus_north_levantine"
        print("Pushing the dataset to the hub ...")
        dataset.push_to_hub(dataset_name, private=True)
        print(f"Dataset {dataset_name} uploaded to the hub")


if __name__ == "__main__":
    prepare_dataset()
