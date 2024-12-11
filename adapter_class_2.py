import json
import os

import numpy as np
import torch
from adapters import AdapterConfig, AdapterTrainer, AutoAdapterModel
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    GPT2TokenizerFast,
    TrainingArguments,
)
from argparse import ArgumentParser
from process_datasets import process_train_dataset, process_test_dataset


def load_dataset_from_config(train_set="baseline", config_path=None, train_path="train.csv", test_path="test.csv"):

    # Load the Yelp Review Full dataset
    # dataset = load_dataset("yelp_review_full")
    # dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})
    ds_config_file = f"configs/dataset-{train_set}.json" if config_path is None else config_path
    dataset_train = process_train_dataset(train_path, ds_config_file)
    dataset_test = process_test_dataset(test_path, ds_config_file)
    dataset = DatasetDict({"train": dataset_train, "test": dataset_test})

    return dataset



def prepare_dataset(
    tokenizer,
    max_length=128,
    dataset: DatasetDict = None,
):

    if "train" not in dataset:
        test_only = True
    else:
        test_only = False

    if test_only:
            print(f"Number of test examples: {len(dataset['test'])}")
    else:
        print(
            f"Number of training examples: {len(dataset['train'])}, number of test examples: {len(dataset['test'])}"
        )

    # Filter and map labels to binary classes
    def preprocess_examples(example):
        if example["label"] in [0, 1, 2]:
            example["label"] = 0  # Negative class
            return example
        elif example["label"] in [3, 4]:
            example["label"] = 1  # Positive class
            return example

    # Apply the preprocessing function to the dataset
    dataset = dataset.map(preprocess_examples)

    # Tokenization function
    def tokenize_function(examples):
        # Tokenize the text with truncation and padding
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # Include the labels
        result["labels"] = examples["label"]
        result["category"] = examples["category"]
        return result

    # Tokenize the datasets
    if not test_only:
        tokenized_train = dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
    else:
        tokenized_train = None

    tokenized_val = dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["test"].column_names,
    )

    # return multiple validation sets
    tokenized_vals = {}
    for cat in ["restaurant", "drinks", "entertainment", "shopping", "housing", "beauty"]:
        tokenized_vals[f"val_{cat}"] = tokenized_val.filter(
            lambda x: x["category"] == cat
        )
    tokenized_vals["val_all"] = tokenized_val
    return tokenized_train, tokenized_vals


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class CustomAdapterTrainer(AdapterTrainer):
    def __init__(self, *args, eval_datasets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_datasets = eval_datasets  # A dictionary of evaluation datasets

    def evaluate(self, eval_dataset=None, **kwargs):
        results = {}
        # Use provided eval_dataset if available
        if eval_dataset is not None:
            result = super().evaluate(eval_dataset=eval_dataset, **kwargs)
            results["eval"] = result
        # Otherwise, evaluate on all datasets in self.eval_datasets
        elif self.eval_datasets is not None:
            for name, dataset in self.eval_datasets.items():
                print(f"\nEvaluating on {name} dataset:")
                result = super().evaluate(eval_dataset=dataset, **kwargs)
                # results[name] = result
                for key, value in result.items():
                    results[f"eval_{name}_{key}"] = value
        else:
            # Default to parent's evaluate method
            results = super().evaluate(**kwargs)
        return results


def main():
    parser = ArgumentParser()
    parser.add_argument("--train_set", type=str, default="baseline")
    parser.add_argument(
        "--model_output_dir", type=str, default="training_outputs/data_100000/"
    )
    parser.add_argument(
        "--eval_output_dir", type=str, default="evaluation_results/data_100000/"
    )
    parser.add_argument("--train_data", type=str, default="data_100000/train_79500.csv")
    parser.add_argument("--test_data", type=str, default="data_100000/test_1500.csv")
    parser.add_argument("--max_epochs", type=int, default=10)
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

    # Add a new adapter configuration
    adapter_config = AdapterConfig.load(
        "pfeiffer",  # Use the Pfeiffer adapter configuration
        reduction_factor=16,
        non_linearity="gelu",
    )

    # Add an adapter and a classification head to the model
    model.add_adapter("yelp_adapter", config=adapter_config)
    model.delete_head("default")
    model.add_classification_head(
        "yelp_head",
        num_labels=2,
    )

    # Activate the adapter and classification head
    model.train_adapter("yelp_adapter")
    model.set_active_adapters("yelp_adapter")
    model.active_head = "yelp_head"

    # Prepare the dataset
    train_dataset, val_dataset = prepare_dataset(
        tokenizer,
        dataset=load_dataset_from_config(
            train_set=args.train_set,
            train_path=args.train_data,
            test_path=args.test_data,
        ),
    )
    # breakpoint()

    # Define the data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"{args.model_output_dir}/gpt2-yelp-adapter_class_2_{args.train_set}",
        overwrite_output_dir=True,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=80,
        per_device_eval_batch_size=200,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        warmup_steps=50,
        learning_rate=1e-3,
        logging_dir="./logs",
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=(
            f"val_{args.train_set}_eval_accuracy"
            if args.train_set != "baseline"
            else f"val_all_eval_accuracy"
            # else "accuracy"
        ),
        greater_is_better=True,
    )

    # Initialize the AdapterTrainer
    trainer = CustomAdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset[f"val_{args.train_set}"] if args.train_set != "baseline" else val_dataset["val_all"],
        eval_datasets=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # if args.train_set != "baseline":
    #     trainer = CustomAdapterTrainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=train_dataset,
    #         eval_dataset=val_dataset[f"val_{args.train_set}"],
    #         eval_datasets=val_dataset,
    #         tokenizer=tokenizer,
    #         data_collator=data_collator,
    #         compute_metrics=compute_metrics,
    #     )
    # else:
    #     trainer = AdapterTrainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=train_dataset,
    #         eval_dataset=val_dataset,
    #         tokenizer=tokenizer,
    #         data_collator=data_collator,
    #         compute_metrics=compute_metrics,
    #     )

    # Start training
    trainer.train()

    # Evaluate the model
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")

    os.makedirs(
        f"{args.eval_output_dir}/{args.train_set}",
        exist_ok=True,
    )
    with open(
        f"{args.eval_output_dir}/{args.train_set}/metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f)

    # Save the adapter and the classification head
    adapter_save_path = f"{args.model_output_dir}/saved_adapters/yelp_adapter_class_2_{args.train_set}"
    os.makedirs(adapter_save_path, exist_ok=True)
    model.save_adapter(adapter_save_path, "yelp_adapter")
    model.save_head(os.path.join(adapter_save_path, "head"), "yelp_head")



if __name__ == "__main__":
    main()
