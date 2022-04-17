from shared import (
    CustomTokens,
    DatasetArguments,
    prepare_datasets,
    load_datasets,
    CustomTrainingArguments,
    get_last_checkpoint,
    train_from_checkpoint
)
from model import ModelArguments
import transformers
import logging
import os
import sys
from datasets import utils as d_utils
from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from dataclasses import dataclass

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version('4.17.0')
require_version('datasets>=1.8.0',
                'To fix: pip install -r requirements.txt')

os.environ['WANDB_DISABLED'] = 'true'

logging.basicConfig()
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)


@dataclass
class Seq2SeqTrainingArguments(CustomTrainingArguments, Seq2SeqTrainingArguments):
    pass


def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    hf_parser = HfArgumentParser((
        ModelArguments,
        DatasetArguments,
        Seq2SeqTrainingArguments
    ))
    model_args, dataset_args, training_args = hf_parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    d_utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    # set_seed(training_args.seed)

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # FP16 https://github.com/huggingface/transformers/issues/9295

    # Works:
    # https://huggingface.co/docs/transformers/model_doc/t5v1.1
    # google/t5-v1_1-small
    # google/t5-v1_1-base
    # google/t5-v1_1-large
    # google/t5-v1_1-xl
    # google/t5-v1_1-xxl

    # https://huggingface.co/docs/transformers/model_doc/t5
    # t5-small
    # t5-base
    # t5-large
    # t5-3b
    # t5-11b

    # allenai/led-base-16384 - https://github.com/huggingface/transformers/issues/9810

    # Further work:
    # Multilingual- https://huggingface.co/docs/transformers/model_doc/mt5

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    raw_datasets = load_datasets(dataset_args)
    # , cache_dir=model_args.cache_dir

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Detecting last checkpoint.
    last_checkpoint = get_last_checkpoint(training_args)

    from model import get_model_tokenizer
    model, tokenizer = get_model_tokenizer(model_args, training_args)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.

    prefix = CustomTokens.EXTRACT_SEGMENTS_PREFIX.value

    PAD_TOKEN_REPLACE_ID = -100

    # https://github.com/huggingface/transformers/issues/5204
    def preprocess_function(examples):
        inputs = examples['text']
        targets = examples['extracted']
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100
        # when we want to ignore padding in the loss.

        model_inputs['labels'] = [
            [(l if l != tokenizer.pad_token_id else PAD_TOKEN_REPLACE_ID)
                for l in label]
            for label in labels['input_ids']
        ]

        return model_inputs

    train_dataset, eval_dataset, predict_dataset = prepare_datasets(
        raw_datasets, dataset_args, training_args, preprocess_function)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=PAD_TOKEN_REPLACE_ID,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Done processing datasets

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    train_result = train_from_checkpoint(
        trainer, last_checkpoint, training_args)

    metrics = train_result.metrics
    max_train_samples = training_args.max_train_samples or len(
        train_dataset)
    metrics['train_samples'] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    kwargs = {'finetuned_from': model_args.model_name_or_path,
              'tasks': 'summarization'}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == '__main__':
    main()
