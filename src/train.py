from preprocess import load_datasets, DatasetArguments
from predict import ClassifierArguments, SPONSOR_MATCH_RE
from shared import CustomTokens, device, GeneralArguments, OutputArguments
from model import ModelArguments, get_model, get_tokenizer
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
import re

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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={'help': 'The number of processes to use for the preprocessing.'},
    )
    # https://github.com/huggingface/transformers/issues/5204
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            'help': 'The maximum total input sequence length after tokenization. Sequences longer '
            'than this will be truncated, sequences shorter will be padded.'
        },
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={
            'help': 'The maximum total sequence length for target text after tokenization. Sequences longer '
            'than this will be truncated, sequences shorter will be padded.'
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            'help': 'The maximum total sequence length for validation target text after tokenization. Sequences longer '
            'than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`.'
            'This argument is also used to override the ``max_length`` param of ``model.generate``, which is used '
            'during ``evaluate`` and ``predict``.'
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            'help': 'Whether to pad all samples to model maximum sentence length. '
            'If False, will pad the samples dynamically when batching to the maximum length in the batch. More '
            'efficient on GPU but very bad for TPU.'
        },
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
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            'help': 'Number of beams to use for evaluation. This argument will be passed to ``model.generate``, '
            'which is used during ``evaluate`` and ``predict``.'
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            'help': 'Whether to ignore the tokens corresponding to padded labels in the loss computation or not.'
        },
    )
    source_prefix: Optional[str] = field(
        default=CustomTokens.EXTRACT_SEGMENTS_PREFIX.value, metadata={
            'help': 'A prefix to add before every source text (useful for T5 models).'}
    )

    # TODO add vectorizer params

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


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
            # max_features=50000
            max_features=10000
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
                # Get matches:
                matches = re_findall(SPONSOR_MATCH_RE, row['extracted'])

                return  # TODO fix

                if not matches:
                    matches = [row['text']]

                for match in matches:
                    train_test_data[ds_type]['X'].append(match)
                    train_test_data[ds_type]['y'].append(row['sponsor'])

        print('Fitting')
        _X_train = vectorizer.fit_transform(train_test_data['train']['X'])
        _X_test = vectorizer.transform(train_test_data['test']['X'])

        y_train = train_test_data['train']['y']
        y_test = train_test_data['test']['y']

        # 2. Create classifier
        classifier = LogisticRegression(max_iter=500)

        # 3. Fit data
        print('fit classifier')
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

        if data_training_args.source_prefix is None and 't5-' in model_args.model_name_or_path:
            logger.warning(
                "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with `--source_prefix 'summarize: ' `"
            )

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

        # Load pretrained model and tokenizer
        tokenizer = get_tokenizer(model_args)
        model = get_model(model_args)
        model.to(device())
        model.resize_token_embeddings(len(tokenizer))

        if model.config.decoder_start_token_id is None:
            raise ValueError(
                'Make sure that `config.decoder_start_token_id` is correctly defined')

        if hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings < data_training_args.max_source_length:
            if model_args.resize_position_embeddings is None:
                logger.warning(
                    f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} to {data_training_args.max_source_length}."
                )
                model.resize_position_embeddings(
                    data_training_args.max_source_length)

            elif model_args.resize_position_embeddings:
                model.resize_position_embeddings(
                    data_training_args.max_source_length)

            else:
                raise ValueError(
                    f'`--max_source_length` is set to {data_training_args.max_source_length}, but the model only has {model.config.max_position_embeddings}'
                    f' position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically '
                    "resize the model's position encodings by passing `--resize_position_embeddings`."
                )

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = raw_datasets['train'].column_names

        # Temporarily set max_target_length for training.
        max_target_length = data_training_args.max_target_length
        padding = 'max_length' if data_training_args.pad_to_max_length else False

        if training_args.label_smoothing_factor > 0 and not hasattr(model, 'prepare_decoder_input_ids_from_labels'):
            logger.warning(
                'label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for'
                f'`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory'
            )

        prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else ''

        # https://github.com/huggingface/transformers/issues/5204
        def preprocess_function(examples):
            inputs = examples['text']
            targets = examples['extracted']
            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(
                inputs, max_length=data_training_args.max_source_length, padding=padding, truncation=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets, max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == 'max_length' and data_training_args.ignore_pad_token_for_loss:
                labels['input_ids'] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels['input_ids']
                ]
            model_inputs['labels'] = labels['input_ids']

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

        max_target_length = data_training_args.val_max_target_length
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
        label_pad_token_id = - \
            100 if data_training_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
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
            print('Saving model')
            trainer.save_model(os.path.join(
                training_args.output_dir, 'checkpoint-latest'))  # TODO use dir
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
