import pickle
import os
from shared import CustomTokens
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='google/t5-v1_1-small',  # t5-small
        metadata={
            'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    # config_name: Optional[str] = field( # TODO remove?
    #     default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'}
    # )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Where to store the pretrained models downloaded from huggingface.co'},
    )
    use_fast_tokenizer: bool = field(  # TODO remove?
        default=True,
        metadata={
            'help': 'Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.'},
    )
    model_revision: str = field(  # TODO remove?
        default='main',
        metadata={
            'help': 'The specific model version to use (can be a branch name, tag name or commit id).'},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            'help': 'Will use the token generated when running `transformers-cli login` (necessary to use this script '
            'with private models).'
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            'help': "Whether to automatically resize the position embeddings if `max_source_length` exceeds the model's position embeddings."
        },
    )


def get_model(model_args, use_cache=True):
    name = model_args.model_name_or_path
    cached_path = f'models/{name}'

    # Model created after tokenizer:
    if use_cache and os.path.exists(os.path.join(cached_path, 'pytorch_model.bin')):
        name = cached_path

    config = AutoConfig.from_pretrained(
        name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        name,
        from_tf='.ckpt' in name,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    return model


def get_tokenizer(model_args, use_cache=True):
    name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path

    cached_path = f'models/{name}'

    if use_cache and os.path.exists(os.path.join(cached_path, 'tokenizer.json')):
        name = cached_path

    tokenizer = AutoTokenizer.from_pretrained(
        name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    CustomTokens.add_custom_tokens(tokenizer)

    return tokenizer


CLASSIFIER_CACHE = {}
def get_classifier_vectorizer(classifier_args, use_cache=True):
    classifier_path = os.path.join(classifier_args.classifier_dir, classifier_args.classifier_file)
    if use_cache and classifier_path in CLASSIFIER_CACHE:
        classifier = CLASSIFIER_CACHE[classifier_path]
    else:
        with open(classifier_path, 'rb') as fp:
            classifier = CLASSIFIER_CACHE[classifier_path] = pickle.load(fp)

    vectorizer_path = os.path.join(classifier_args.classifier_dir, classifier_args.vectorizer_file)
    if use_cache and vectorizer_path in CLASSIFIER_CACHE:
        vectorizer = CLASSIFIER_CACHE[vectorizer_path]
    else:
        with open(vectorizer_path, 'rb') as fp:
            vectorizer = CLASSIFIER_CACHE[vectorizer_path] = pickle.load(fp)

    return classifier, vectorizer
