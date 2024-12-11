# llm-adapter

## Train
```bash
python adapter_class_2.py --train_set baseline --max_epochs 20
```

explaination of the arguments:
- `--train_set`: the dataset to train on. The corresponding dataset cofinguration is in `configs/dataset-{train_set}.py`
- `--max_epochs`: the number of epochs to train the model for

## Inference
```bash
python inference.py --model_dir "training_outputs/data_100000/saved_adapters/yelp_adapter_class_2_baseline/" --test_data "data_100000/test/baseline.csv" --test_output_dir "test_results/"
```

explaination of the arguments:
- `--model_dir`: the directory where the trained adapter (with head) is saved
- `--test_data`: the path to the test data file
- `--test_output_dir`: the directory where the test results will be saved

## Datasets
### data_100000
#### Train
- baseline: 5000 samples in all. Sample numbers for each category are listed below:

    |               | Restaurant | Drinks | Shopping | Entertainment | Housing | Beauty |
|---------------|------------|--------|----------|---------------|---------|--------|
| Amount        | 1000       | 1000   | 1000     | 1000          | 500     | 500    |

- restaurant: 5000 samples in Restaurant category.
- drinks: 5000 samples in Drinks category.
- shopping: 5000 samples in Shopping category.
- entertainment: 5000 samples in Entertainment category.
- housing: 2500 samples in Housing category.
- beauty: 2500 samples in Beauty category.

#### Test
1500 samples in all. 250 samples for each category.
