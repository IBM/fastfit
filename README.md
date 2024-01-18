## Running the Training Script

Our package provides a convenient command-line tool `train_fastfit` to train text classification models. This tool comes with a variety of configurable parameters to customize your training process.

### Prerequisites

Before running the training script, ensure you have Python installed along with our package and its dependencies. If you haven't already installed our package, you can do so using pip:

```bash
pip install fast-fit
```

### Usage

To run the training script with custom configurations, use the `train_fastfit` command followed by the necessary arguments similar to huggingface training args with few additions relevant for fast-fit. Here's the general syntax for the command:

```bash
train_fastfit --model_name_or_path [MODEL_NAME] --overwrite_output_dir [BOOLEAN] --report_to [REPORT_SETTING] --label_column_name [LABEL_COLUMN_NAME] --text_column_name [TEXT_COLUMN_NAME] --max_steps [MAX_STEPS] --dataloader_drop_last [BOOLEAN] --per_device_train_batch_size [BATCH_SIZE] --per_device_eval_batch_size [BATCH_SIZE] --evaluation_strategy [EVAL_STRATEGY] --num_repeats [NUM_REPEATS] --save_strategy [SAVE_STRATEGY] --proj_dim [PROJECTION_DIMENSION] --fp16 [BOOLEAN] --learning_rate [LEARNING_RATE] --optim [OPTIMIZER] --do_train [BOOLEAN] --do_eval [BOOLEAN] --max_text_length [MAX_TEXT_LENGTH] --output_dir [OUTPUT_DIR] --train_file [TRAIN_FILE] --validation_file [VALIDATION_FILE]
```

Replace the bracketed terms with your desired settings. Here's an explanation of each parameter:

- `model_name_or_path`: Identifier or path of the model (e.g., 'roberta-large').
- `overwrite_output_dir`: Whether to overwrite the output directory (`True` or `False`).
- `report_to`: Destination for logging or reporting (e.g., 'none').
- `label_column_name`: Column name for labels in your dataset.
- `text_column_name`: Column name for text data in your dataset.
- `max_steps`: Maximum number of training steps (e.g., 1500).
- `dataloader_drop_last`: Whether to drop the last incomplete batch (`True` or `False`).
- `per_device_train_batch_size`: Batch size per device during training.
- `per_device_eval_batch_size`: Batch size per device during evaluation.
- `evaluation_strategy`: Strategy for evaluation (e.g., 'no', 'steps').
- `num_repeats`: Number of noisy embeddings repetitions.
- `save_strategy`: Model saving strategy (e.g., 'no', 'epoch').
- `proj_dim`: Projection dimension for model-specific projections (e.g., 128).
- `fp16`: Use of mixed precision training (`True` or `False`).
- `learning_rate`: Learning rate for optimizer (e.g., 1e-5).
- `optim`: Choice of optimizer (e.g., 'adamw_hf').
- `do_train`: Flag to run training (`True` or `False`).
- `do_eval`: Flag to run evaluation (`True` or `False`).
- `max_text_length`: Maximum text sequence length.
- `output_dir`: Output directory for model checkpoints and results.
- `train_file`: Path to training data file.
- `validation_file`: Path to validation data file.

### Example Command

Here's an example of how to use the `run_train` command with specific settings:

```bash
train_fastfit --model_name_or_path roberta-large --overwrite_output_dir True --report_to none --label_column_name label --text_column_name text --max_steps 1500 --dataloader_drop_last True --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --evaluation_strategy no --num_repeats 4 --save_strategy no --proj_dim 128 --learning_rate 1e-5 --optim adamw_hf --do_train True --do_eval True --max_text_length 256 --output_dir ./output --train_file $TRAIN_FILE --validation_file $DEV_FILE
```

### Output

Upon execution, `train_fastfit` will start the training process based on your parameters and output the results, including logs and model checkpoints, to the designated directory.

## Training with python
You can simply run it with your python

```python
from datasets import load_dataset
from fastfit import FastFitTrainer

# Load a dataset from the Hugging Face Hub
dataset = load_dataset("SetFit/sst2")

# Down sample the train data for 10-shot training
dataset["train"] = dataset["train"].shuffle().select(range(10))

trainer = FastFitTrainer(
    model_name_or_path="roberta-large",
    overwrite_output_dir=True,
    report_to="none",
    label_column_name="label_text",
    text_column_name="text",
    max_steps=1500,
    dataloader_drop_last=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="no",
    num_repeats=4,
    save_strategy="no",
    proj_dim=128,
    learning_rate=1e-5,
    optim="adamw_hf",
    max_text_length=256,
    output_dir="./output",
    dataset=dataset,
)

model = trainer.train()
results = trainer.evaluate()
test_results = trainer.test()
```
