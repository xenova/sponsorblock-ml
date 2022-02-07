from datasets import load_dataset
from preprocess import DatasetArguments
from predict import ClassifierArguments, SEGMENT_MATCH_RE, CATEGORIES
from shared import CustomTokens, GeneralArguments, OutputArguments
from model import ModelArguments
import transformers
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import pickle
from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import re_findall

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version('4.13.0.dev0')
require_version('datasets>=1.8.0',
                'To fix: pip install -r requirements.txt')

os.environ['WANDB_DISABLED'] = 'true'


logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)


def load_datasets(dataset_args):

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
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={'help': 'The number of processes to use for the preprocessing.'},
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            'help': 'For debugging purposes or quicker training, truncate the number of training examples to this value if set.'
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            'help': 'For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.'
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            'help': 'For debugging purposes or quicker training, truncate the number of prediction examples to this value if set.'
        },
    )


@dataclass
class SequenceTrainingArguments(OutputArguments, Seq2SeqTrainingArguments):
    seed: Optional[int] = GeneralArguments.__dataclass_fields__['seed']

    num_train_epochs: float = field(
        default=1, metadata={'help': 'Total number of training epochs to perform.'})

    save_steps: int = field(default=5000, metadata={
                            'help': 'Save checkpoint every X updates steps.'})
    eval_steps: int = field(default=5000, metadata={
                            'help': 'Run an evaluation every X steps.'})
    logging_steps: int = field(default=5000, metadata={
                               'help': 'Log every X updates steps.'})

    skip_train_transformer: bool = field(default=False, metadata={
        'help': 'Whether to skip training the transformer.'})
    train_classifier: bool = field(default=False, metadata={
        'help': 'Whether to run training on the 2nd phase (classifier).'})

    # do_eval: bool = field(default=False, metadata={
    #                       'help': 'Whether to run eval on the dev set.'})
    do_predict: bool = field(default=False, metadata={
                             'help': 'Whether to run predictions on the test set.'})

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


def main():
    # reset()

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    hf_parser = HfArgumentParser((
        ModelArguments,
        DatasetArguments,
        DataTrainingArguments,
        SequenceTrainingArguments,
        ClassifierArguments
    ))
    model_args, dataset_args, data_training_args, training_args, classifier_args = hf_parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
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
    if training_args.skip_train_transformer and not training_args.train_classifier:
        print('Nothing to do. Exiting')
        return

    raw_datasets = load_datasets(dataset_args)
    # , cache_dir=model_args.cache_dir

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.train_classifier:
        print('Train classifier')
        # 1. Vectorize raw data to pass into classifier
        # CountVectorizer TfidfVectorizer
        # TfidfVectorizer - better (comb of CountVectorizer)
        vectorizer = TfidfVectorizer(  # CountVectorizer
            # lowercase=False,
            # stop_words='english',  # TODO optimise stop words?
            # stop_words=stop_words,

            ngram_range=(1, 2),  # best so far
            # max_features=8000  # remove for higher accuracy?
            max_features=20000
            # max_features=10000
            # max_features=1000
        )

        train_test_data = {
            'train': {
                'X': [],
                'y': []
            },
            'test': {
                'X': [],
                'y': []
            }
        }

        print('Splitting')
        for ds_type in train_test_data:
            dataset = raw_datasets[ds_type]

            for row in dataset:
                matches = re_findall(SEGMENT_MATCH_RE, row['extracted'])
                if matches:
                    for match in matches:
                        train_test_data[ds_type]['X'].append(match['text'])

                        class_index = CATEGORIES.index(match['category'])
                        train_test_data[ds_type]['y'].append(class_index)

                else:
                    train_test_data[ds_type]['X'].append(row['text'])
                    train_test_data[ds_type]['y'].append(0)

        print('Fitting')
        _X_train = vectorizer.fit_transform(train_test_data['train']['X'])
        _X_test = vectorizer.transform(train_test_data['test']['X'])

        y_train = train_test_data['train']['y']
        y_test = train_test_data['test']['y']

        # 2. Create classifier
        classifier = LogisticRegression(max_iter=2000, class_weight='balanced')

        # 3. Fit data
        print('Fit classifier')
        classifier.fit(_X_train, y_train)

        # 4. Measure accuracy
        accuracy = classifier.score(_X_test, y_test)

        print(f'[LogisticRegression] Accuracy percent:',
              round(accuracy*100, 3))

        # 5. Save classifier and vectorizer
        with open(os.path.join(classifier_args.classifier_dir, classifier_args.classifier_file), 'wb') as fp:
            pickle.dump(classifier, fp)

        with open(os.path.join(classifier_args.classifier_dir, classifier_args.vectorizer_file), 'wb') as fp:
            pickle.dump(vectorizer, fp)

    if not training_args.skip_train_transformer:
        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.'
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
                )

        from model import get_model_tokenizer
        model, tokenizer = get_model_tokenizer(
            model_args.model_name_or_path, model_args.cache_dir)
        # max_tokenizer_length = model.config.d_model

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = raw_datasets['train'].column_names

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

        def prepare_dataset(dataset, desc):
            return dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_training_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not dataset_args.overwrite_cache,
                desc=desc,  # tokenizing train dataset
            )
        # train_dataset # TODO shuffle?

        # if training_args.do_train:
        if 'train' not in raw_datasets:  # TODO do checks above?
            raise ValueError('Train dataset missing')
        train_dataset = raw_datasets['train']
        if data_training_args.max_train_samples is not None:
            train_dataset = train_dataset.select(
                range(data_training_args.max_train_samples))
        with training_args.main_process_first(desc='train dataset map pre-processing'):
            train_dataset = prepare_dataset(
                train_dataset, desc='Running tokenizer on train dataset')

        if 'validation' not in raw_datasets:
            raise ValueError('Validation dataset missing')
        eval_dataset = raw_datasets['validation']
        if data_training_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(
                range(data_training_args.max_eval_samples))
        with training_args.main_process_first(desc='validation dataset map pre-processing'):
            eval_dataset = prepare_dataset(
                eval_dataset, desc='Running tokenizer on validation dataset')

        if 'test' not in raw_datasets:
            raise ValueError('Test dataset missing')
        predict_dataset = raw_datasets['test']
        if data_training_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(
                range(data_training_args.max_predict_samples))
        with training_args.main_process_first(desc='prediction dataset map pre-processing'):
            predict_dataset = prepare_dataset(
                predict_dataset, desc='Running tokenizer on prediction dataset')

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
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        try:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
        except KeyboardInterrupt:
            # TODO add option to save model on interrupt?
            # print('Saving model')
            # trainer.save_model(os.path.join(
            #     training_args.output_dir, 'checkpoint-latest'))  # TODO use dir
            raise

        metrics = train_result.metrics
        max_train_samples = data_training_args.max_train_samples or len(
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
