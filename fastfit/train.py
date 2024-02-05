import logging
import os
import random
import sys
import json
import math

from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Optional

import torch
import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from transformers.integrations import INTEGRATION_TO_CALLBACK
from .modeling import ConfigArguments
from .modeling import FastFitTrainable, FastFitConfig

INTEGRATION_TO_CALLBACK["clearml"] = INTEGRATION_TO_CALLBACK["tensorboard"]

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

task_to_keys = {"custom": None}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    metric_name: Optional[str] = field(
        default="accuracy",
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )
    custom_goal_acc: Optional[float] = field(
        default=None,
        metadata={"help": "If set, save the model every this number of steps."},
    )
    text_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the labels."
        },
    )
    max_text_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for text."
        },
    )
    max_label_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for label."
        },
    )
    pre_train: Optional[bool] = field(
        default=False, metadata={"help": "The path to the pretrained model."}
    )
    added_tokens_per_label: Optional[int] = field(
        default=None,
        metadata={"help": "The number of added tokens to add to every class."},
    )
    added_tokens_mask_factor: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "How much of the added tokens should be consisted of mask tokens embedding."
        },
    )
    added_tokens_tfidf_factor: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "How much of the added tokens should be consisted of tfidf tokens embedding."
        },
    )
    pad_query_with_mask: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to pad the query with the mask token."},
    )
    pad_doc_with_mask: Optional[bool] = field(
        default=False, metadata={"help": "Whether to pad the docs with the mask token."}
    )
    doc_mapper: Optional[str] = field(
        default=None, metadata={"help": "The source for mapping docs to augmented docs"}
    )
    doc_mapper_type: Optional[str] = field(
        default="file", metadata={"help": "The type of doc mapper"}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(task_to_keys.keys())
                )
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )
    load_from_FastFit: bool = field(
        default=False,
        metadata={"help": "Will load the model from the trained model directory."},
    )


parser = HfArgumentParser(
    (ConfigArguments, ModelArguments, DataTrainingArguments, TrainingArguments)
)


def tfidf(docs):  # docs is list of tokenized documents
    all_terms = set([t for doc in docs for t in doc])
    tf = [{t: (v / len(doc)) for t, v in Counter(doc).items()} for doc in docs]
    df = Counter({t: len([d for d in docs if t in d]) for t in all_terms})
    idf = {t: math.log(len(docs) / v) for t, v in df.items()}
    tfidf = [{t: (tf[i][t] * idf[t]) for t in doc} for i, doc in enumerate(docs)]
    return tfidf


def get_args(args_dict=None):
    if args_dict is None:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            config_args, model_args, data_args, training_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1])
            )
        else:
            (
                config_args,
                model_args,
                data_args,
                training_args,
            ) = parser.parse_args_into_dataclasses()
    else:
        config_args, model_args, data_args, training_args = parser.parse_dict(args_dict)

    return config_args, model_args, data_args, training_args


class FastFitTrainer:
    def has_custom_dataset(self):
        return (
            self._dataset is not None
            or self._train_dataset is not None
            or self._test_datast is not None
            or self._validation_dataset is not None
        )

    def set_args(self, args_dict=None):
        if args_dict is None or len(args_dict) == 0:
            if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
                # If we pass only one argument to the script and it's the path to a json file,
                # let's parse it to get our arguments.
                (
                    config_args,
                    model_args,
                    data_args,
                    training_args,
                ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
            else:
                (
                    config_args,
                    model_args,
                    data_args,
                    training_args,
                ) = parser.parse_args_into_dataclasses()
        else:
            args_dict["do_train"] = True
            args_dict["do_eval"] = True
            args_dict["do_predict"] = True
            if self.has_custom_dataset():
                args_dict["task_name"] = "custom"
            config_args, model_args, data_args, training_args = parser.parse_dict(
                args_dict
            )

        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        self.config_args = config_args

    def set_logger(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        log_level = self.training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
            + f"distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {self.training_args}")
        self.logger = logger

    def set_last_checkpoint(self):
        last_checkpoint = None
        if (
            os.path.isdir(self.training_args.output_dir)
            and self.training_args.do_train
            and not self.training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if (
                last_checkpoint is None
                and len(os.listdir(self.training_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({self.training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                last_checkpoint is not None
                and self.training_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        self.checkpoint = checkpoint
        return checkpoint

    def set_seed(self):
        set_seed(self.training_args.seed)

    def set_data(
        self,
    ):
        if self._dataset is not None:
            raw_datasets = self._dataset
        elif (
            self._train_dataset is not None
            or self._validation_dataset is not None
            or self._test_dataset is not None
        ):
            raw_datasets = datasets.DatasetDict()
            if self._train_dataset is not None:
                raw_datasets["train"] = self._train_dataset
            if self._validation_dataset is not None:
                raw_datasets["validation"] = self._validation_dataset
            if self._test_dataset is not None:
                raw_datasets["test"] = self._test_dataset
        elif self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                cache_dir=self.model_args.cache_dir,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from your local files.
            # CSV/JSON training and evaluation files are needed.
            data_files = {
                "train": self.data_args.train_file,
                "validation": self.data_args.validation_file,
            }

            # Get the test dataset: you can provide your own CSV/JSON test file (see below)
            # when you use `do_predict` without specifying a GLUE benchmark task.
            if self.training_args.do_predict:
                if self.data_args.test_file is not None:
                    train_extension = self.data_args.train_file.split(".")[-1]
                    test_extension = self.data_args.test_file.split(".")[-1]
                    assert (
                        test_extension == train_extension
                    ), "`test_file` should have the same extension (csv or json) as `train_file`."
                    data_files["test"] = self.data_args.test_file
                else:
                    raise ValueError(
                        "Need either a GLUE task or a test file for `do_predict`."
                    )

            for key in data_files.keys():
                logger.info(f"load a local file for {key}: {data_files[key]}")

            if self.data_args.train_file.endswith(".csv"):
                # Loading a dataset from local csv files
                raw_datasets = load_dataset(
                    "csv",
                    data_files=data_files,
                    cache_dir=self.model_args.cache_dir,
                    use_auth_token=True if self.model_args.use_auth_token else None,
                )
            else:
                # Loading a dataset from local json files
                raw_datasets = load_dataset(
                    "json",
                    data_files=data_files,
                    cache_dir=self.model_args.cache_dir,
                    use_auth_token=True if self.model_args.use_auth_token else None,
                )

        self.is_regression = raw_datasets["train"].features[
            self.data_args.label_column_name
        ].dtype in ["float32", "float64"]
        if self.is_regression:
            self.num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            self.labels = raw_datasets["train"].unique(self.data_args.label_column_name)
            if self.training_args.do_eval:
                self.labels = sorted(
                    list(
                        set(
                            self.labels
                            + raw_datasets["validation"].unique(
                                self.data_args.label_column_name
                            )
                        )
                    )
                )
            if self.training_args.do_predict:
                self.labels = sorted(
                    list(
                        set(
                            self.labels
                            + raw_datasets["test"].unique(
                                self.data_args.label_column_name
                            )
                        )
                    )
                )
            self.labels.sort()  # Let's sort it for determinism
            self.num_labels = len(self.labels)
            self.labels_mapping = {
                label: label.replace("_", " ") if isinstance(label, str) else label
                for label in self.labels
            }

        self.raw_datasets = raw_datasets

        if self.data_args.doc_mapper is not None:
            if self.data_args.doc_mapper_type == "file":
                with open(self.data_args.doc_mapper, "r") as f:
                    self.doc_mapper = json.load(f)
                    assert all(
                        [label in self.doc_mapper for label in self.labels]
                    ), f"labels {self.labels} not in doc_mapper {self.doc_mapper.keys()}"
                    self.labels_mapping = {
                        label: self.doc_mapper[label] for label in self.labels
                    }
            else:
                raise ValueError(
                    f"doc_mapper_type {self.data_args.doc_mapper_type} not supported"
                )

    def set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name
            if self.model_args.tokenizer_name
            else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        if self.data_args.added_tokens_per_label is not None:
            if self.data_args.added_tokens_per_label > 0:
                docs = defaultdict(list)
                for example in self.raw_datasets["train"]:
                    tokenized = self.tokenizer.tokenize(
                        example[self.data_args.text_column_name],
                        add_special_tokens=False,
                    )
                    docs[example[self.data_args.label_column_name]].extend(tokenized)
                tfidf_scores = tfidf(docs.values())
                tfidf_tokens = [
                    [
                        t[0]
                        for t in Counter(doc_scores).most_common(
                            self.data_args.added_tokens_per_label
                        )
                    ]
                    for doc_scores in tfidf_scores
                ]
                self.tfidf_tokens = [
                    item for sublist in tfidf_tokens for item in sublist
                ]

                self.added_tokens = [
                    [
                        f"<LABEL_{j}_TOKEN_{i}>"
                        for i in range(self.data_args.added_tokens_per_label)
                    ]
                    for j in range(self.num_labels)
                ]
                self.labels_mapping = {
                    self.labels[j]: self.labels_mapping[self.labels[j]]
                    + "".join(self.added_tokens[j])
                    for j in range(self.num_labels)
                }
                self.all_added_tokens = [
                    item for sublist in self.added_tokens for item in sublist
                ]
                self.tokenizer.add_tokens(self.all_added_tokens)

    def set_model(self):
        config_kwargs = vars(self.config_args)

        if self.model_args.load_from_FastFit:
            config = FastFitConfig.from_pretrained(
                pretrained_model_name_or_path=self.model_args.model_name_or_path,
                clf_dim=self.num_labels,
                **config_kwargs,
            )
            model = FastFitTrainable.from_pretrained(
                pretrained_model_name_or_path=self.model_args.model_name_or_path,
            )
        else:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.model_args.model_name_or_path,
            )
            config = FastFitConfig.from_encoder_config(
                config,
                clf_dim=self.num_labels,
                **config_kwargs,
            )
            model = FastFitTrainable.from_encoder_pretrained(
                encoder_pretrained_model_name_or_path=self.model_args.model_name_or_path,
                config=config,
            )

        if self.data_args.added_tokens_per_label is not None:
            if self.data_args.added_tokens_per_label > 0:
                embeddings = model.resize_token_embeddings(len(self.tokenizer))
                if (
                    self.data_args.added_tokens_mask_factor > 0
                    or self.data_args.added_tokens_tfidf_factor > 0
                ):
                    old_factor = max(
                        0,
                        1
                        - self.data_args.added_tokens_mask_factor
                        - self.data_args.added_tokens_tfidf_factor,
                    )
                    mask_token_embed = embeddings.weight[self.tokenizer.mask_token_id]
                    for added_token_id, tfidf_token_id in zip(
                        self.tokenizer.convert_tokens_to_ids(self.all_added_tokens),
                        self.tokenizer.convert_tokens_to_ids(self.tfidf_tokens),
                    ):
                        tfidf_token_embed = embeddings.weight[tfidf_token_id]
                        with torch.no_grad():
                            embeddings.weight[added_token_id] = (
                                mask_token_embed
                                * self.data_args.added_tokens_mask_factor
                                + tfidf_token_embed
                                * self.data_args.added_tokens_tfidf_factor
                                + embeddings.weight[added_token_id] * old_factor
                            )
                    logger.info("Initialized added tokens from mask token")

        self.model = model
        self.config = config

    def preprocess_data(self):
        # Padding strategy
        if self.data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        self.label_to_id = None
        if (
            self.model.config.label2id
            != PretrainedConfig(num_labels=self.num_labels).label2id
            and self.data_args.task_name is not None
            and not self.is_regression
        ):
            # Some have all caps in their config, some don't.
            self.label_name_to_id = {
                k.lower(): v for k, v in self.model.config.label2id.items()
            }
            if list(sorted(self.label_name_to_id.keys())) == list(sorted(self.labels)):
                self.label_to_id = {
                    i: int(self.label_name_to_id[self.labels[i]])
                    for i in range(self.num_labels)
                }
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(self.label_name_to_id.keys()))}, dataset labels: {list(sorted(self.labels))}."
                    "\nIgnoring the model labels as a result.",
                )
        elif self.data_args.task_name is None and not self.is_regression:
            self.label_to_id = {v: i for i, v in enumerate(self.labels)}

        if self.label_to_id is not None:
            self.model.config.label2id = self.label_to_id
            self.model.config.id2label = {
                id: label for label, id in self.config.label2id.items()
            }

        # infer max length for both query and document
        with self.training_args.main_process_first(
            desc="infer max length for both query and document"
        ):

            def get_lengths(examples):
                batch_labels = examples[self.data_args.label_column_name]
                batch_labels = [self.labels_mapping[label] for label in batch_labels]
                batch_texts = examples[self.data_args.text_column_name]
                result = {}
                result["lens_texts"] = [
                    len(ids)
                    for ids in self.tokenizer(batch_texts, truncation=True)["input_ids"]
                ]
                result["lens_labels"] = [
                    len(ids)
                    for ids in self.tokenizer(batch_labels, truncation=True)[
                        "input_ids"
                    ]
                ]
                return result

            lens_datasets = self.raw_datasets.map(
                get_lengths,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset to infer max length for both query and document",
            )

            max_lens_texts = max(lens_datasets["train"]["lens_texts"])
            max_lens_labels = max(lens_datasets["train"]["lens_labels"])

            if self.data_args.pad_query_with_mask:
                max_length_texts = min(
                    self.data_args.max_text_length, self.tokenizer.model_max_length
                )
            else:
                max_length_texts = min(
                    self.data_args.max_text_length,
                    max_lens_texts,
                    self.tokenizer.model_max_length,
                )

            if self.data_args.pad_doc_with_mask:
                max_length_labels = min(
                    self.data_args.max_label_length, self.tokenizer.model_max_length
                )
            else:
                max_length_labels = min(
                    self.data_args.max_label_length,
                    max_lens_labels,
                    self.tokenizer.model_max_length,
                )

        tokenized_labels = self.tokenizer(
            [self.labels_mapping[label] for label in self.labels],
            max_length=max_length_labels,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if self.data_args.pad_doc_with_mask:
            tokenized_labels["input_ids"][
                tokenized_labels["input_ids"].eq(self.tokenizer.pad_token_id)
            ] = self.tokenizer.mask_token_id
            tokenized_labels["attention_mask"][:, :] = 1

        self.model.set_documetns(
            (tokenized_labels["input_ids"], tokenized_labels["attention_mask"])
        )

        def preprocess_function(examples):
            batch_labels = examples[self.data_args.label_column_name]
            batch_texts = examples[self.data_args.text_column_name]

            result = self.tokenizer(
                batch_texts,
                padding=padding,
                max_length=max_length_texts,
                truncation=True,
            )

            if self.data_args.pad_query_with_mask:
                assert padding == "max_length"
                result["input_ids"] = [
                    [
                        tok
                        if tok != self.tokenizer.pad_token_id
                        else self.tokenizer.mask_token_id
                        for tok in tokens
                    ]
                    for tokens in result["input_ids"]
                ]
                result["attention_mask"] = [
                    [1 for _ in tokens] for tokens in result["attention_mask"]
                ]

            result["query_input_ids"] = result.get("input_ids")
            result["query_attention_mask"] = result.pop("attention_mask")

            batch_labels = [self.labels_mapping[t] for t in batch_labels]
            doc = self.tokenizer(
                batch_labels,
                padding=padding,
                max_length=max_length_labels,
                truncation=True,
            )

            if self.data_args.pad_doc_with_mask:
                assert padding == "max_length"
                doc["input_ids"] = [
                    [
                        tok
                        if tok != self.tokenizer.pad_token_id
                        else self.tokenizer.mask_token_id
                        for tok in tokens
                    ]
                    for tokens in doc["input_ids"]
                ]
                doc["attention_mask"] = [
                    [1 for _ in tokens] for tokens in doc["attention_mask"]
                ]

            result["doc_input_ids"] = doc.pop("input_ids")
            result["doc_attention_mask"] = doc.pop("attention_mask")

            if (
                self.label_to_id is not None
                and self.data_args.label_column_name in examples
            ):
                result["label"] = [
                    (self.label_to_id[label] if label != -1 else -1)
                    for label in examples[self.data_args.label_column_name]
                ]

            return result

        with self.training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = self.raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        if self.training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            self.train_dataset = raw_datasets["train"]
            if self.data_args.max_train_samples is not None:
                max_train_samples = min(
                    len(self.train_dataset), self.data_args.max_train_samples
                )
                self.train_dataset = self.train_dataset.select(range(max_train_samples))

        if self.training_args.do_eval:
            self.eval_dataset = raw_datasets["validation"]
            # delete the doc_input_ids and doc_attention_mask from the eval dataset
            # eval_dataset = eval_dataset.remove_columns(['doc_input_ids', 'doc_attention_mask'])
            if self.data_args.max_eval_samples is not None:
                max_eval_samples = min(
                    len(self.eval_dataset), self.data_args.max_eval_samples
                )
                self.eval_dataset = self.eval_dataset.select(range(max_eval_samples))
            # create a eval dataset for inference that contain every text with every possible label

        if (
            self.training_args.do_predict
            or self.data_args.task_name is not None
            or self.data_args.test_file is not None
        ):
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            self.predict_dataset = raw_datasets[
                "test_matched" if self.data_args.task_name == "mnli" else "test"
            ]
            if self.data_args.max_predict_samples is not None:
                max_predict_samples = min(
                    len(self.predict_dataset), self.data_args.max_predict_samples
                )
                self.predict_dataset = self.predict_dataset.select(
                    range(max_predict_samples)
                )

        # Log a few random samples from the training set:
        if self.training_args.do_train:
            for index in random.sample(range(len(self.train_dataset)), 3):
                logger.info(
                    f"Sample {index} of the training set: {self.train_dataset[index]}."
                )

    def set_trainer(self):
        metric = load_metric(self.data_args.metric_name)

        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def compute_metrics(p: EvalPrediction):
            predictions = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            predictions = (
                np.squeeze(predictions)
                if self.is_regression
                else np.argmax(predictions, axis=1)
            )
            references = p.label_ids
            return metric.compute(predictions=predictions, references=references)

        # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
        # we already did the padding.
        if self.data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif self.training_args.fp16:
            data_collator = DataCollatorWithPadding(
                self.tokenizer, pad_to_multiple_of=8
            )
        else:
            data_collator = None

        # Initialize our Trainer
        self.tokenizer.model_input_names = [
            "query_input_ids",
            "query_attention_mask",
            "doc_input_ids",
            "doc_attention_mask",
        ]

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset if self.training_args.do_train else None,
            eval_dataset=self.eval_dataset if self.training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

    def __init__(
        self,
        dataset=None,
        train_dataset=None,
        validation_dataset=None,
        test_dataset=None,
        **kwargs,
    ):
        self._dataset = dataset
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset
        self._test_dataset = test_dataset
        self.set_args(kwargs)
        self.set_logger()
        self.set_last_checkpoint()
        self.set_seed()
        self.set_data()
        self.set_tokenizer()
        self.set_model()
        self.preprocess_data()
        self.set_trainer()

    def train(self):
        # Training
        if self.training_args.do_train:
            train_result = self.trainer.train(
                resume_from_checkpoint=self.checkpoint,
                ignore_keys_for_eval={"doc_input_ids", "doc_attention_mask", "labels"},
            )
            metrics = train_result.metrics
            max_train_samples = (
                self.data_args.max_train_samples
                if self.data_args.max_train_samples is not None
                else len(self.train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

            if self.training_args.save_strategy != "no":
                self.trainer.save_model()  # Saves the tokenizer too for easy upload

            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()

        return self.model

    def evaluate(self):
        if self.training_args.do_eval:
            self.logger.info("*** Evaluate ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [self.data_args.task_name]
            eval_datasets = [self.eval_dataset]
            if self.data_args.task_name == "mnli":
                tasks.append("mnli-mm")
                eval_datasets.append(self.raw_datasets["validation_mismatched"])
                combined = {}

            for eval_dataset, task in zip(eval_datasets, tasks):
                metrics = self.trainer.evaluate(eval_dataset=eval_dataset)

                max_eval_samples = (
                    self.data_args.max_eval_samples
                    if self.data_args.max_eval_samples is not None
                    else len(eval_dataset)
                )
                metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

                if task == "mnli-mm":
                    metrics = {k + "_mm": v for k, v in metrics.items()}
                if task is not None and "mnli" in task:
                    combined.update(metrics)

                self.trainer.log_metrics("eval", metrics)
                self.trainer.save_metrics(
                    "eval", combined if task is not None and "mnli" in task else metrics
                )

            return metrics

    def test(self):
        metrics = None
        if self.training_args.do_predict:
            self.logger.info("*** Predict ***")
            self.logger.info("*** Evaluate ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [self.data_args.task_name]
            predict_datasets = [self.predict_dataset]
            if self.data_args.task_name == "mnli":
                tasks.append("mnli-mm")
                self.eval_datasets.append(self.raw_datasets["validation_mismatched"])
                combined = {}

            for predict_dataset, task in zip(predict_datasets, tasks):
                metrics = self.trainer.evaluate(eval_dataset=predict_dataset)

                max_predict_samples = (
                    self.data_args.max_eval_samples
                    if self.data_args.max_eval_samples is not None
                    else len(predict_dataset)
                )
                metrics["test_samples"] = min(max_predict_samples, len(predict_dataset))

                if task == "mnli-mm":
                    metrics = {k + "_mm": v for k, v in metrics.items()}
                if task is not None and "mnli" in task:
                    combined.update(metrics)

                self.trainer.log_metrics("test", metrics)
                self.trainer.save_metrics(
                    "test", combined if task is not None and "mnli" in task else metrics
                )

        return metrics

    def push_to_hub(self):
        kwargs = {
            "finetuned_from": self.model_args.model_name_or_path,
            "tasks": "text-classification",
        }
        if self.data_args.task_name is not None:
            kwargs["language"] = "en"
            kwargs["dataset_tags"] = "glue"
            kwargs["dataset_args"] = self.data_args.task_name
            kwargs["dataset"] = f"GLUE {self.data_args.task_name.upper()}"

        if self.training_args.push_to_hub:
            self.trainer.push_to_hub(**kwargs)


def main():
    trainer = FastFitTrainer()
    trainer.train()
    trainer.evaluate()
    trainer.test()
    trainer.push_to_hub()

    # else:
    #     trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

__OUTPUT_DIR_ARG__ = "output_dir"
__OUTPUTS__ = ["all_results.json"]
__ARGPARSER__ = parser
