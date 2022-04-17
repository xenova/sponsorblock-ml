from transformers.trainer_utils import get_last_checkpoint as glc
from transformers import Seq2SeqTrainingArguments, TrainingArguments
import os
from utils import re_findall
import logging
import sys
from datasets import load_dataset
import re
import gc
from time import time_ns
import random
import numpy as np
import torch
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

CATEGORIES = [None, 'SPONSOR', 'SELFPROMO', 'INTERACTION']

ACTION_OPTIONS = ['skip', 'mute', 'full']

CATGEGORY_OPTIONS = {
    'SPONSOR': 'Sponsor',
    'SELFPROMO': 'Self/unpaid promo',
    'INTERACTION': 'Interaction reminder',
}

START_SEGMENT_TEMPLATE = 'START_{}_TOKEN'
END_SEGMENT_TEMPLATE = 'END_{}_TOKEN'


class CustomTokens(Enum):
    EXTRACT_SEGMENTS_PREFIX = 'EXTRACT_SEGMENTS: '

    # Preprocessing tokens
    URL = 'URL_TOKEN'
    HYPHENATED_URL = 'HYPHENATED_URL_TOKEN'
    NUMBER_PERCENTAGE = 'NUMBER_PERCENTAGE_TOKEN'
    NUMBER = 'NUMBER_TOKEN'

    SHORT_HYPHENATED = 'SHORT_HYPHENATED_TOKEN'
    LONG_WORD = 'LONG_WORD_TOKEN'

    # Custom YouTube tokens
    MUSIC = '[Music]'
    APPLAUSE = '[Applause]'
    LAUGHTER = '[Laughter]'

    PROFANITY = 'PROFANITY_TOKEN'

    # Segment tokens
    NO_SEGMENT = 'NO_SEGMENT_TOKEN'

    START_SPONSOR = START_SEGMENT_TEMPLATE.format('SPONSOR')
    END_SPONSOR = END_SEGMENT_TEMPLATE.format('SPONSOR')

    START_SELFPROMO = START_SEGMENT_TEMPLATE.format('SELFPROMO')
    END_SELFPROMO = END_SEGMENT_TEMPLATE.format('SELFPROMO')

    START_INTERACTION = START_SEGMENT_TEMPLATE.format('INTERACTION')
    END_INTERACTION = END_SEGMENT_TEMPLATE.format('INTERACTION')

    BETWEEN_SEGMENTS = 'BETWEEN_SEGMENTS_TOKEN'

    @classmethod
    def custom_tokens(cls):
        return [e.value for e in cls]

    @classmethod
    def add_custom_tokens(cls, tokenizer):
        tokenizer.add_tokens(cls.custom_tokens())


_SEGMENT_START = START_SEGMENT_TEMPLATE.format(r'(?P<category>\w+)')
_SEGMENT_END = END_SEGMENT_TEMPLATE.format(r'\w+')
SEGMENT_MATCH_RE = fr'{_SEGMENT_START}\s*(?P<text>.*?)\s*(?:{_SEGMENT_END}|$)'


def extract_sponsor_matches_from_text(text):
    if CustomTokens.NO_SEGMENT.value in text:
        return []
    else:
        return re_findall(SEGMENT_MATCH_RE, text)


def extract_sponsor_matches(texts):
    return list(map(extract_sponsor_matches_from_text, texts))


@dataclass
class DatasetArguments:
    data_dir: Optional[str] = field(
        default='data',
        metadata={
            'help': 'The directory which stores train, test and/or validation data.'
        },
    )
    processed_file: Optional[str] = field(
        default='segments.json',
        metadata={
            'help': 'Processed data file'
        },
    )
    processed_database: Optional[str] = field(
        default='processed_database.json',
        metadata={
            'help': 'Processed database file'
        },
    )

    overwrite_cache: bool = field(
        default=False, metadata={'help': 'Overwrite the cached training and evaluation sets'}
    )

    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Where to store the cached datasets'
        },
    )

    train_file: Optional[str] = field(
        default='train.json', metadata={'help': 'The input training data file (a jsonlines file).'}
    )
    validation_file: Optional[str] = field(
        default='valid.json',
        metadata={
            'help': 'An optional input evaluation data file to evaluate the metrics on (a jsonlines file).'
        },
    )
    test_file: Optional[str] = field(
        default='test.json',
        metadata={
            'help': 'An optional input test data file to evaluate the metrics on (a jsonlines file).'
        },
    )

    c_train_file: Optional[str] = field(
        default='c_train.json', metadata={'help': 'The input training data file (a jsonlines file).'}
    )
    c_validation_file: Optional[str] = field(
        default='c_valid.json',
        metadata={
            'help': 'An optional input evaluation data file to evaluate the metrics on (a jsonlines file).'
        },
    )
    c_test_file: Optional[str] = field(
        default='c_test.json',
        metadata={
            'help': 'An optional input test data file to evaluate the metrics on (a jsonlines file).'
        },
    )

    def __post_init__(self):
        if self.train_file is None or self.validation_file is None:
            raise ValueError(
                'Need either a dataset name or a training/validation file.')

        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class OutputArguments:

    output_dir: str = field(
        default='out',
        metadata={
            'help': 'The output directory where the model predictions and checkpoints will be written to and read from.'
        },
    )
    checkpoint: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Choose the checkpoint/model to train from or test with. Defaults to the latest checkpoint found in `output_dir`.'
        },
    )
    models_dir: str = field(
        default='models',
        metadata={
            'help': 'The output directory where the model predictions and checkpoints will be written to and read from.'
        },
    )
    # classifier_dir: str = field(
    #     default='out',
    #     metadata={
    #         'help': 'The output directory where the model predictions and checkpoints will be written to and read from.'
    #     },
    # )


def seed_factory():
    return time_ns() % (2**32 - 1)


@dataclass
class GeneralArguments:
    seed: Optional[int] = field(default_factory=seed_factory, metadata={
        'help': 'Set seed for deterministic training and testing. By default, it uses the current time (results in essentially random results).'
    })
    no_cuda: bool = field(default=False, metadata={
                          'help': 'Do not use CUDA even when it is available'})

    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)


def seconds_to_time(seconds, remove_leading_zeroes=False):
    fractional = round(seconds % 1, 3)
    fractional = '' if fractional == 0 else str(fractional)[1:]
    h, remainder = divmod(abs(int(seconds)), 3600)
    m, s = divmod(remainder, 60)
    hms = f'{h:02}:{m:02}:{s:02}'
    if remove_leading_zeroes:
        hms = re.sub(r'^0(?:0:0?)?', '', hms)
    return f"{'-' if seconds < 0 else ''}{hms}{fractional}"


def reset():
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    gc.collect()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))


def load_datasets(dataset_args: DatasetArguments):

    print('Reading datasets')
    data_files = {}

    if dataset_args.train_file is not None:
        data_files['train'] = os.path.join(
            dataset_args.data_dir, dataset_args.train_file)
    if dataset_args.validation_file is not None:
        data_files['validation'] = os.path.join(
            dataset_args.data_dir, dataset_args.validation_file)
    if dataset_args.test_file is not None:
        data_files['test'] = os.path.join(
            dataset_args.data_dir, dataset_args.test_file)

    return load_dataset('json', data_files=data_files, cache_dir=dataset_args.dataset_cache_dir)


@dataclass
class AdditionalTrainingArguments:
    seed: Optional[int] = GeneralArguments.__dataclass_fields__['seed']

    num_train_epochs: float = field(
        default=1, metadata={'help': 'Total number of training epochs to perform.'})

    save_steps: int = field(default=5000, metadata={
                            'help': 'Save checkpoint every X updates steps.'})
    eval_steps: int = field(default=25000, metadata={
                            'help': 'Run an evaluation every X steps.'})
    logging_steps: int = field(default=5000, metadata={
                               'help': 'Log every X updates steps.'})

    # do_eval: bool = field(default=False, metadata={
    #                       'help': 'Whether to run eval on the dev set.'})
    # do_predict: bool = field(default=False, metadata={
    #                          'help': 'Whether to run predictions on the test set.'})

    per_device_train_batch_size: int = field(
        default=4, metadata={'help': 'Batch size per GPU/TPU core/CPU for training.'}
    )
    per_device_eval_batch_size: int = field(
        default=4, metadata={'help': 'Batch size per GPU/TPU core/CPU for evaluation.'}
    )

    # report_to: Optional[List[str]] = field(
    #     default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    # )
    evaluation_strategy: str = field(
        default='steps',
        metadata={
            'help': 'The evaluation strategy to use.',
            'choices': ['no', 'steps', 'epoch']
        },
    )

    # evaluation_strategy (:obj:`str` or :class:`~transformers.trainer_utils.IntervalStrategy`, `optional`, defaults to :obj:`"no"`):
    # The evaluation strategy to adopt during training. Possible values are:

    #     * :obj:`"no"`: No evaluation is done during training.
    #     * :obj:`"steps"`: Evaluation is done (and logged) every :obj:`eval_steps`.
    #     * :obj:`"epoch"`: Evaluation is done at the end of each epoch.

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={'help': 'The number of processes to use for the preprocessing.'},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )


@dataclass
class CustomTrainingArguments(OutputArguments, AdditionalTrainingArguments):
    pass


logging.basicConfig()
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)


def get_last_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = glc(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    return last_checkpoint


def train_from_checkpoint(trainer, last_checkpoint, training_args):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model()  # Saves the tokenizer too for easy upload

    return train_result


def prepare_datasets(raw_datasets, dataset_args: DatasetArguments, training_args: CustomTrainingArguments, preprocess_function):

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not dataset_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if 'train' not in raw_datasets:
        raise ValueError('Train dataset missing')
    train_dataset = raw_datasets['train']
    if training_args.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(training_args.max_train_samples))

    if 'validation' not in raw_datasets:
        raise ValueError('Validation dataset missing')
    eval_dataset = raw_datasets['validation']
    if training_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(
            range(training_args.max_eval_samples))

    if 'test' not in raw_datasets:
        raise ValueError('Test dataset missing')
    predict_dataset = raw_datasets['test']
    if training_args.max_predict_samples is not None:
        predict_dataset = predict_dataset.select(
            range(training_args.max_predict_samples))

    return train_dataset, eval_dataset, predict_dataset
