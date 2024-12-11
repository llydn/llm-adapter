import json
import os

import numpy as np
import torch
from adapters import AdapterTrainer, AutoAdapterModel
from datasets import load_dataset
from transformers import (
    DataCollatorWithPadding,
    GPT2TokenizerFast,
)
from argparse import ArgumentParser
from adapter_class_2 import CustomAdapterTrainer, prepare_dataset, compute_metrics


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="training_outputs/data_100000/saved_adapters/yelp_adapter_class_2_baseline/",
    )
    parser.add_argument("--test_output_dir", type=str, default="test_results/")
    parser.add_argument(
        "--test_data", type=str, default="data_100000/test/baseline.csv"
    )
    args = parser.parse_args()

    # Check if CUDA is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer and model
    model_name = "gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    # Load GPT-2 model with adapter support
    model = AutoAdapterModel.from_pretrained(model_name)
    model.to(device)

    model.load_adapter(args.model_dir, set_active=True, load_as="yelp_adapter")
    model.delete_head("default")
    model.load_head(os.path.join(args.model_dir, "head"), set_active=True)


    # Prepare the dataset
    dataset = load_dataset("csv", data_files={"test": args.test_data})
    _ , test_dataset = prepare_dataset(
        tokenizer,
        dataset=dataset,
    )

    # Define the data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize the AdapterTrainer
    trainer = CustomAdapterTrainer(
        model=model,
        eval_dataset=(
            test_dataset["val_all"]
        ),
        eval_datasets=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


    # Evaluate the model
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")

    os.makedirs(
        f"{args.test_output_dir}",
        exist_ok=True,
    )
    with open(
        f"{args.test_output_dir}/metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f)



if __name__ == "__main__":
    main()
