from functools import lru_cache
import pickle
import os
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
            'help': 'Path to pretrained model or model identifier from huggingface.co/models'
        }
    )
    # config_name: Optional[str] = field( # TODO remove?
    #     default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'}
    # )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={
            'help': 'Pretrained tokenizer name or path if not the same as model_name'
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Where to store the pretrained models downloaded from huggingface.co'
        },
    )
    use_fast_tokenizer: bool = field(  # TODO remove?
        default=True,
        metadata={
            'help': 'Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.'
        },
    )
    model_revision: str = field(  # TODO remove?
        default='main',
        metadata={
            'help': 'The specific model version to use (can be a branch name, tag name or commit id).'
        },
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


@lru_cache(maxsize=None)
def get_classifier_vectorizer(classifier_args):
    classifier_path = os.path.join(
        classifier_args.classifier_dir, classifier_args.classifier_file)
    with open(classifier_path, 'rb') as fp:
        classifier = pickle.load(fp)

    vectorizer_path = os.path.join(
        classifier_args.classifier_dir, classifier_args.vectorizer_file)
    with open(vectorizer_path, 'rb') as fp:
        vectorizer = pickle.load(fp)

    return classifier, vectorizer
