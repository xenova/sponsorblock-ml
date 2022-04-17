
""" Finetuning the library models for sequence classification."""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import datasets
import numpy as np

import transformers
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from shared import (
    CATEGORIES,
    DatasetArguments,
    prepare_datasets,
    load_datasets,
    CustomTrainingArguments,
    train_from_checkpoint,
    get_last_checkpoint
)
from model import get_model_tokenizer, ModelArguments

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version('4.17.0')
require_version('datasets>=1.8.0', 'To fix: pip install -r requirements.txt')

os.environ['WANDB_DISABLED'] = 'true'

logger = logging.getLogger(__name__)


@dataclass
class ClassifierTrainingArguments(CustomTrainingArguments, TrainingArguments):
    pass


@dataclass
class ClassifierDatasetArguments(DatasetArguments):
    train_file: Optional[str] = DatasetArguments.__dataclass_fields__[
        'c_train_file']
    validation_file: Optional[str] = DatasetArguments.__dataclass_fields__[
        'c_validation_file']
    test_file: Optional[str] = DatasetArguments.__dataclass_fields__[
        'c_test_file']


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    hf_parser = HfArgumentParser((
        ModelArguments,
        ClassifierDatasetArguments,
        ClassifierTrainingArguments
    ))
    model_args, dataset_args, training_args = hf_parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint.
    last_checkpoint = get_last_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    raw_datasets = load_datasets(dataset_args)

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    config_args = {
        'num_labels': len(CATEGORIES),
        'id2label': {k: str(v).upper() for k, v in enumerate(CATEGORIES)},
        'label2id': {str(v).upper(): k for k, v in enumerate(CATEGORIES)}
    }
    model, tokenizer = get_model_tokenizer(
        model_args, training_args, config_args=config_args, model_type='classifier')

    if training_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f'The max_seq_length passed ({training_args.max_seq_length}) is larger than the maximum length for the'
            f'model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.'
        )
    max_seq_length = min(training_args.max_seq_length,
                         tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(
            examples['text'], padding='max_length', max_length=max_seq_length, truncation=True)
        result['label'] = examples['label']
        return result

    train_dataset, eval_dataset, predict_dataset = prepare_datasets(
        raw_datasets, dataset_args, training_args, preprocess_function)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(
            p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {'accuracy': (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if training_args.fp16:
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    train_result = train_from_checkpoint(
        trainer, last_checkpoint, training_args)

    metrics = train_result.metrics
    max_train_samples = (
        training_args.max_train_samples if training_args.max_train_samples is not None else len(
            train_dataset)
    )
    metrics['train_samples'] = min(max_train_samples, len(train_dataset))

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    kwargs = {'finetuned_from': model_args.model_name_or_path,
              'tasks': 'text-classification'}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == '__main__':
    main()
