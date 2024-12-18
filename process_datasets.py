import os
import json
from datasets import load_dataset, concatenate_datasets


def process_train_dataset(dataset_path, config_path, rewrite=False):
    # Load configuration from the JSON file
    with open(config_path, "r") as f:
        config = json.load(f)

    if rewrite == False and config["data_save_path"]["train"] is not None:
        train_path = config["data_save_path"]["train"]
        if os.path.exists(train_path):
            print(f"Loading dataset from {train_path}")
            dataset = load_dataset("csv", data_files=train_path, split="train")
            return dataset

    dataset = load_dataset("csv", data_files=dataset_path, split="train").shuffle(
        seed=config["random_seed"]
    )
    categories = config["categories"]
    category_configs = config["category_configs"]

    # Filter dataset by categories
    dataset_splits = [
        dataset.filter(lambda example: example["category"] == cat) for cat in categories
    ]

    for i, cat in enumerate(categories):
        cat_config = category_configs.get(cat, {})
        ds = dataset_splits[i]
        # Apply select range if specified
        if "select_num" in cat_config:
            n = cat_config["select_num"]
            ds = ds.select(range(n))

        # Shard if num_shards and index are specified
        if "num_shards" in cat_config:
            ds = ds.shard(num_shards=cat_config["num_shards"], index=0)

        if "max_num" in cat_config and len(ds) > cat_config["max_num"]:
            ds = ds.select(range(cat_config["max_num"]))

        dataset_splits[i] = ds
        print(f"Number of training examples in {cat}: {len(ds)}")

    final_dataset = concatenate_datasets(dataset_splits)

    # Save the dataset
    if config["data_save_path"]["train"] is not None:
        os.makedirs(os.path.dirname(config["data_save_path"]["train"]), exist_ok=True)
        final_dataset.to_csv(config["data_save_path"]["train"])
    return final_dataset


def process_test_dataset(
    dataset_path,
    config_path=None,
    categories=["restaurant", "drinks", "entertainment", "shopping"],
    size_per_cat=50,
    seed=42,
    save_path=None,
    rewrite=False,
    get_remainings=False,
):
    # Load configuration from the JSON file
    if config_path is not None:
        with open(config_path, "r") as f:
            config = json.load(f)
        if rewrite == False and config["data_save_path"]["test"] is not None:
            test_path = config["data_save_path"]["test"]
            if os.path.exists(test_path):
                print(f"Loading dataset from {test_path}")
                dataset = load_dataset("csv", data_files=test_path, split="train")
                return dataset
        if "test_size_per_category" in config:
            size_per_cat = config["test_size_per_category"]

    dataset = load_dataset("csv", data_files=dataset_path, split="train").shuffle(
        seed=seed
    )

    # Filter dataset by categories
    categories = config["categories"] if config_path is not None else categories
    dataset_splits = [
        dataset.filter(lambda example: example["category"] == cat) for cat in categories
    ]
    dataset_remainings = []

    for i, cat in enumerate(categories):
        ds = dataset_splits[i]
        total_examples = len(ds)
        if total_examples < size_per_cat:
            print(f"Warning: Number of examples in {cat} is less than {size_per_cat}")
            ds = ds.select(range(total_examples))
            ds_remain = None
        else:
            ds_remain = ds.select(range(size_per_cat, total_examples))
            ds = ds.select(range(size_per_cat))
        dataset_splits[i] = ds
        print(f"Number of test examples in {cat}: {len(ds)}")

        if ds_remain is not None:
            dataset_remainings.append(ds_remain)
            print(f"Number of remaining examples in {cat}: {len(ds_remain)}")

    final_dataset = concatenate_datasets(dataset_splits)
    final_dataset_remainings = concatenate_datasets(dataset_remainings)

    # Save the dataset
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        final_dataset.to_csv(save_path)

    if config_path is not None:
        if config["data_save_path"]["test"] != save_path:
            os.makedirs(
                os.path.dirname(config["data_save_path"]["test"]), exist_ok=True
            )
            final_dataset.to_csv(config["data_save_path"]["test"])

    if get_remainings:
        return final_dataset, final_dataset_remainings
    return final_dataset


if __name__ == "__main__":
    final_dataset = process_train_dataset("data_100000/train_79500.csv", "configs/dataset-baseline.json", rewrite=True)
    # test_data, remainig_data = process_test_dataset("classification_results_100000.csv", "configs/dataset-baseline.json", rewrite=True, get_remainings=True)
    # remainig_data.to_csv("data_100000/test_remainings.csv")
