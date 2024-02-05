## Running the Training Script

Our package provides a convenient command-line tool `train_fastfit` to train text classification models. This tool comes with a variety of configurable parameters to customize your training process.

### Prerequisites

Before running the training script, ensure you have Python installed along with our package and its dependencies. If you haven't already installed our package, you can do so using pip:

```bash
pip install fast-fit
```

### Usage

To run the training script with custom configurations, use the `train_fastfit` command followed by the necessary arguments similar to huggingface training args with few additions relevant for fast-fit.

### Example Command

Here's an example of how to use the `run_train` command with specific settings:

```bash
train_fastfit \
    --model_name_or_path "roberta-base" \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --output_dir ./tmp/try \
    --overwrite_output_dir \
    --report_to none \
    --label_column_name label\
    --text_column_name text \
    --num_train_epochs 40 \
    --dataloader_drop_last true \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --evaluation_strategy steps \
    --max_text_length 128 \
    --logging_steps 100 \
    --dataloader_drop_last=False \
    --num_repeats 4 \
    --save_strategy no \
    --optim adafactor \
    --clf_loss_factor 0.1 \
    --do_train \
    --fp16 \
    --projection_dim 128
```

### Output

Upon execution, `train_fastfit` will start the training process based on your parameters and output the results, including logs and model checkpoints, to the designated directory.

## Training with python
You can simply run it with your python

```python
from datasets import load_dataset
from fastfit import FastFitTrainer, sample_dataset

# Load a dataset from the Hugging Face Hub
dataset = load_dataset("mteb/banking77")
dataset["validation"] = dataset["test"]

# Down sample the train data for 5-shot training
dataset["train"] = sample_dataset(dataset["train"], label_column="label", num_samples_per_label=5)

trainer = FastFitTrainer(
    model_name_or_path="roberta-base",
    overwrite_output_dir=True,
    label_column_name="label",
    text_column_name="text",
    num_train_epochs=40,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    max_text_length=128,
    dataloader_drop_last=False,
    num_repeats=4,
    optim="adafactor",
    clf_loss_factor=0.1,
    fp16=True,
    dataset=dataset,
)

model = trainer.train()
results = trainer.evaluate()
test_results = trainer.test()

model.save_pretrained("fast-fit")
```
Then you can use the model for inference
```python
from fastfit import FastFit
from transformers import AutoTokenizer, pipeline

model = FastFit.from_pretrained("fast-fit")
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

print(classifier("I love this package!"))
```

## All avialble parameters:
**Optional Arguments:**

- `-h, --help`: Show this help message and exit.
- `--num_repeats NUM_REPEATS`: The number of times to repeat the queries and docs in every batch. (default: 1)
- `--proj_dim PROJ_DIM`: The dimension of the projection layer. (default: 128)
- `--clf_loss_factor CLF_LOSS_FACTOR`: The factor to scale the classification loss. (default: 0.1)
- `--pretrain_mode [PRETRAIN_MODE]`: Whether to do pre-training. (default: False)
- `--inference_type INFERENCE_TYPE`: The inference type to be used. (default: sim)
- `--rep_tokens REP_TOKENS`: The tokens to use for representation when calculating the similarity in training and inference. (default: all)
- `--length_norm [LENGTH_NORM]`: Whether to normalize by length while considering pad (default: False)
- `--mlm_factor MLM_FACTOR`: The factor to scale the MLM loss. (default: 0.0)
- `--mask_prob MASK_PROB`: The probability of masking a token. (default: 0.0)
- `--model_name_or_path MODEL_NAME_OR_PATH`: Path to pretrained model or model identifier from huggingface.co/models (default: None)
- `--config_name CONFIG_NAME`: Pretrained config name or path if not the same as model_name (default: None)
- `--tokenizer_name TOKENIZER_NAME`: Pretrained tokenizer name or path if not the same as model_name (default: None)
- `--cache_dir CACHE_DIR`: Where do you want to store the pretrained models downloaded from huggingface.co (default: None)
- `--use_fast_tokenizer [USE_FAST_TOKENIZER]`: Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. (default: True)
- `--no_use_fast_tokenizer`: Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. (default: False)
- `--model_revision MODEL_REVISION`: The specific model version to use (can be a branch name, tag name, or commit id). (default: main)
- `--use_auth_token [USE_AUTH_TOKEN]`: Will use the token generated when running `transformers-cli login` (necessary to use this script with private models). (default: False)
- `--ignore_mismatched_sizes [IGNORE_MISMATCHED_SIZES]`: Will enable to load a pretrained model whose head dimensions are different. (default: False)
- `--load_from_FastFit [LOAD_FROM_FASTFIT]`: Will load the model from the trained model directory. (default: False)
- `--task_name TASK_NAME`: The name of the task to train on: custom (default: None)
- `--metric_name METRIC_NAME`: The name of the task to train on: custom (default: accuracy)
- `--dataset_name DATASET_NAME`: The name of the dataset to use (via the datasets library). (default: None)
- `--dataset_config_name DATASET_CONFIG_NAME`: The configuration name of the dataset to use (via the datasets library). (default: None)
- `--max_seq_length MAX_SEQ_LENGTH`: The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. (default: 128)
- `--overwrite_cache [OVERWRITE_CACHE]`: Overwrite the cached preprocessed datasets or not. (default: False)
- `--pad_to_max_length [PAD_TO_MAX_LENGTH]`: Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch. (default: True)
- `--no_pad_to_max_length`: Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch. (default: False)
- `--max_train_samples MAX_TRAIN_SAMPLES`: For debugging purposes or quicker training, truncate the number of training examples to this value if set. (default: None)
- `--max_eval_samples MAX_EVAL_SAMPLES`: For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set. (default: None)
- `--max_predict_samples MAX_PREDICT_SAMPLES`: For debugging purposes or quicker training, truncate the number of prediction examples to this value if set. (default: None)
- `--train_file TRAIN_FILE`: A csv or a json file containing the training data. (default: None)
- `--validation_file VALIDATION_FILE`: A csv or a json file containing the validation data. (default: None)
- `--test_file TEST_FILE`: A csv or a json file containing the test data. (default: None)
- `--custom_goal_acc CUSTOM_GOAL_ACC`: If set, save the model every this number of steps. (default: None)
- `--text_column_name TEXT_COLUMN_NAME`: The name of the column in the datasets containing the full texts (for summarization). (default: None)
- `--label_column_name LABEL_COLUMN_NAME`: The name of the column in the datasets containing the labels. (default: None)
- `--max_text_length MAX_TEXT_LENGTH`: The maximum total input sequence length after tokenization for text. (default: 32)
- `--max_label_length MAX_LABEL_LENGTH`: The maximum total input sequence length after tokenization for label. (default: 32)
- `--pre_train [PRE_TRAIN]`: The path to the pretrained model. (default: False)
- `--added_tokens_per_label ADDED_TOKENS_PER_LABEL`: The number of added tokens to add to every class. (default: None)
- `--added_tokens_mask_factor ADDED_TOKENS_MASK_FACTOR`: How much of the added tokens should be consisted of mask tokens embedding. (default: 0.0)
- `--added_tokens_tfidf_factor ADDED_TOKENS_TFIDF_FACTOR`: How much of the added tokens should be consisted of tfidf tokens embedding. (default: 0.0)
- `--pad_query_with_mask [PAD_QUERY_WITH_MASK]`: Whether to pad the query with the mask token. (default: False)
- `--pad_doc_with_mask [PAD_DOC_WITH_MASK]`: Whether to pad the docs with the mask token. (default: False)
- `--doc_mapper DOC_MAPPER`: The source for mapping docs to augmented docs (default: None)
- `--doc_mapper_type DOC_MAPPER_TYPE`: The type of doc mapper (default: file)
- `--output_dir OUTPUT_DIR`: The output directory where the model predictions and checkpoints will be written. (default: None)
- `--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]`: Overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory. (default: False)
- `--do_train [DO_TRAIN]`: Whether to run training. (default: False)
- `--do_eval [DO_EVAL]`: Whether to run eval on the dev set. (default: False)
- `--do_predict [DO_PREDICT]`: Whether to run predictions on the test set. (default: False)
- `--evaluation_strategy {no,steps,epoch}`: The evaluation strategy to use. (default: no)
- `--prediction_loss_only [PREDICTION_LOSS_ONLY]`: When performing evaluation and predictions, only returns the loss. (default: False)
- `--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE`: Batch size per GPU/TPU core/CPU for training. (default: 8)
- `--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE`: Batch size per GPU/TPU core/CPU for evaluation. (default: 8)
- `--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE`: Deprecated, the use of `--per_device_train_batch_size` is preferred. Batch size per GPU/TPU core/CPU for training. (default: None)
- `--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE`: Deprecated, the use of `--per_device_eval_batch_size` is preferred. Batch size per GPU/TPU core/CPU for evaluation. (default: None)
- `--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS`: Number of updates steps to accumulate before performing a backward/update pass. (default: 1)
- `--eval_accumulation_steps EVAL_ACCUMULATION_STEPS`: Number of predictions steps to accumulate before moving the tensors to the CPU. (default: None)
- `--eval_delay EVAL_DELAY`: Number of epochs or steps to wait for before the first evaluation can be performed, depending on the evaluation_strategy. (default: 0)
- `--learning_rate LEARNING_RATE`: The initial learning rate for AdamW. (default: 5e-05)
- `--weight_decay WEIGHT_DECAY`: Weight decay for AdamW if we apply some. (default: 0.0)
- `--adam_beta1 ADAM_BETA1`: Beta1 for AdamW optimizer (default: 0.9)
- `--adam_beta2 ADAM_BETA2`: Beta2 for AdamW optimizer (default: 0.999)
- `--adam_epsilon ADAM_EPSILON`: Epsilon for AdamW optimizer. (default: 1e-08)
- `--max_grad_norm MAX_GRAD_NORM`: Max gradient norm. (default: 1.0)
- `--num_train_epochs NUM_TRAIN_EPOCHS`: Total number of training epochs to perform. (default: 3.0)
- `--max_steps MAX_STEPS`: If > 0: set the total number of training steps to perform. Override num_train_epochs. (default: -1)
- `--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}`: The scheduler type to use. (default: linear)
- `--warmup_ratio WARMUP_RATIO`: Linear warmup over warmup_ratio fraction of total steps. (default: 0.0)
- `--warmup_steps WARMUP_STEPS`: Linear warmup over warmup_steps. (default: 0)
- `--log_level {debug,info,warning,error,critical,passive}`: Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error', and 'critical', plus a 'passive' level which doesn't set anything and lets the application set the level. Defaults to 'passive'. (default: passive)
- `--log_level_replica {debug,info,warning,error,critical,passive}`: Logger log level to use on replica nodes. Same choices and defaults as `log_level` (default: passive)
- `--log_on_each_node [LOG_ON_EACH_NODE]`: When doing a multinode distributed training, whether to log once per node or just once on the main node. (default: True)
- `--no_log_on_each_node`: When doing a multinode distributed training, whether to log once per node or just once on the main node. (default: False)
- `--logging_dir LOGGING_DIR`: Tensorboard log dir. (default: None)
- `--logging_strategy {no,steps,epoch}`: The logging strategy to use. (default: steps)
- `--logging_first_step [LOGGING_FIRST_STEP]`: Log the first global_step (default: False)
- `--logging_steps LOGGING_STEPS`: Log every X updates steps. (default: 500)
- `--logging_nan_inf_filter [LOGGING_NAN_INF_FILTER]`: Filter nan and inf losses for logging. (default: True)
- `--no_logging_nan_inf_filter`: Filter nan and inf losses for logging. (default: False)
- `--save_strategy {no,steps,epoch}`: The checkpoint save strategy to use. (default: steps)
- `--save_steps SAVE_STEPS`: Save checkpoint every X updates steps. (default: 500)
- `--save_total_limit SAVE_TOTAL_LIMIT
