import gc
from time import time_ns
import random
import numpy as np
import torch
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum


class CustomTokens(Enum):
    URL = 'URL_TOKEN'
    HYPHENATED_URL = 'HYPHENATED_URL_TOKEN'
    NUMBER_PERCENTAGE = 'NUMBER_PERCENTAGE_TOKEN'
    NUMBER = 'NUMBER_TOKEN'

    START_SPONSOR = 'START_SPONSOR'
    END_SPONSOR = 'END_SPONSOR'
    NO_SPONSOR = 'NO_SPONSOR_FOUND'

    SHORT_HYPHENATED = 'SHORT_HYPHENATED_TOKEN'
    LONG_WORD = 'LONG_WORD_TOKEN'

    # Custom YouTube tokens
    MUSIC = '[Music]'
    APPLAUSE = '[Applause]'
    LAUGHTER = '[Laughter]'

    PROFANITY_RAW = '[ __ ]'  # How YouTube transcribes profanity
    PROFANITY_CONVERTED = '*****'  # Safer version for tokenizing
    PROFANITY = 'PROFANITY_TOKEN'

    @classmethod
    def custom_tokens(cls):
        return [e.value for e in cls]

    @classmethod
    def add_custom_tokens(cls, tokenizer):
        tokenizer.add_tokens(cls.custom_tokens())


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

    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reset():
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    gc.collect()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
