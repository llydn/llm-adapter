import json
import os

import numpy as np
import torch
from adapters import AdapterConfig, AdapterTrainer, AutoAdapterModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (DataCollatorWithPadding, EvalPrediction,
                          GPT2TokenizerFast, TrainingArguments)


def prepare_dataset(tokenizer, max_length=128):
    # Load the Yelp Review Full dataset
    # dataset = load_dataset("yelp_review_full")
    dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

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
        return result

    # Tokenize the datasets
    tokenized_train = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    tokenized_val = dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["test"].column_names,
    )

    return tokenized_train, tokenized_val


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
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
        num_labels=5,  # Yelp Review Full has labels from 0 to 4 (5 classes)
        # id2label={0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"},
    )

    # Activate the adapter and classification head
    model.train_adapter("yelp_adapter")
    model.set_active_adapters("yelp_adapter")
    model.active_head = "yelp_head"

    # Prepare the dataset
    train_dataset, val_dataset = prepare_dataset(tokenizer)

    # Define the data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2-yelp-adapter_class_5",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=80,
        per_device_eval_batch_size=200,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=1000,
        warmup_steps=500,
        learning_rate=1e-3,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Initialize the AdapterTrainer
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # Evaluate the model
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")

    # Save the adapter and the classification head
    adapter_save_path = "./saved_adapters/yelp_adapter"
    os.makedirs(adapter_save_path, exist_ok=True)
    model.save_adapter(adapter_save_path, "yelp_adapter")
    model.save_head(os.path.join(adapter_save_path, "head"),"yelp_head")

    # Save the adapter configuration
    adapter_config_dict = adapter_config.to_dict()
    with open(os.path.join(adapter_save_path, "adapter_config.json"), "w") as f:
        json.dump(adapter_config_dict, f)


if __name__ == "__main__":
    main()
