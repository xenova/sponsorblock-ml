from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments
from shared import CustomTokens, GeneralArguments
from dataclasses import dataclass, field
from typing import Optional, Union
import torch
import classify
import base64
import re
import requests
import json
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)

# Public innertube key (b64 encoded so that it is not incorrectly flagged)
INNERTUBE_KEY = base64.b64decode(
    b'QUl6YVN5QU9fRkoyU2xxVThRNFNURUhMR0NpbHdfWTlfMTFxY1c4').decode()

YT_CONTEXT = {
    'client': {
        'userAgent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36,gzip(gfe)',
        'clientName': 'WEB',
        'clientVersion': '2.20211221.00.00',
    }
}
_YT_INITIAL_DATA_RE = r'(?:window\s*\[\s*["\']ytInitialData["\']\s*\]|ytInitialData)\s*=\s*({.+?})\s*;\s*(?:var\s+meta|</script|\n)'


def get_all_channel_vids(channel_id):
    continuation = None
    while True:
        if continuation is None:
            params = {'list': channel_id.replace('UC', 'UU', 1)}
            response = requests.get(
                'https://www.youtube.com/playlist', params=params)
            items = json.loads(re.search(_YT_INITIAL_DATA_RE, response.text).group(1))['contents']['twoColumnBrowseResultsRenderer']['tabs'][0]['tabRenderer']['content'][
                'sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0]['playlistVideoListRenderer']['contents']
        else:
            params = {'key': INNERTUBE_KEY}
            data = {
                'context': YT_CONTEXT,
                'continuation': continuation
            }
            response = requests.post(
                'https://www.youtube.com/youtubei/v1/browse', params=params, json=data)
            items = response.json()[
                'onResponseReceivedActions'][0]['appendContinuationItemsAction']['continuationItems']

        new_token = None
        for vid in items:
            info = vid.get('playlistVideoRenderer')
            if info:
                yield info['videoId']
                continue

            info = vid.get('continuationItemRenderer')
            if info:
                new_token = info['continuationEndpoint']['continuationCommand']['token']

        if new_token is None:
            break
        continuation = new_token



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            'help': 'Path to pretrained model or model identifier from huggingface.co/models'
        }
    )

    cache_dir: Optional[str] = field(
        default='models',
        metadata={
            'help': 'Where to store the pretrained models downloaded from huggingface.co'
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            'help': 'Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.'
        },
    )
    model_revision: str = field(
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

import itertools
from errors import InferenceException, ModelLoadError

@dataclass
class InferenceArguments(ModelArguments):

    model_name_or_path: str = field(
        default='Xenova/sponsorblock-small',
        metadata={
            'help': 'Path to pretrained model used for prediction'
        }
    )
    classifier_model_name_or_path: str = field(
        default='EColi/SB_Classifier',
        metadata={
            'help': 'Use a pretrained classifier'
        }
    )

    max_videos: Optional[int] = field(
        default=None,
        metadata={
            'help': 'The number of videos to test on'
        }
    )
    start_index: int = field(default=None, metadata={
        'help': 'Video to start the evaluation at.'})
    channel_id: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Used to evaluate a channel'
        }
    )
    video_ids: str = field(
        default_factory=lambda: [],
        metadata={
            'nargs': '+'
        }
    )

    output_as_json: bool = field(default=False, metadata={
                                 'help': 'Output evaluations as JSON'})

    min_probability: float = field(
        default=0.5, metadata={'help': 'Remove all predictions whose classification probability is below this threshold.'})

    def __post_init__(self):

        self.video_ids = list(map(str.strip, self.video_ids))

        if any(len(video_id) != 11 for video_id in self.video_ids):
            raise InferenceException('Invalid video IDs (length not 11)')

        if self.channel_id is not None:
            start = self.start_index or 0
            end = None if self.max_videos is None else start + self.max_videos

            channel_video_ids = list(itertools.islice(get_all_channel_vids(
                self.channel_id), start, end))
            logger.info(
                f'Found {len(channel_video_ids)} for channel {self.channel_id}')

            self.video_ids += channel_video_ids



def get_model_tokenizer_classifier(inference_args: InferenceArguments, general_args: GeneralArguments):

    original_path = inference_args.model_name_or_path

    # Load main model and tokenizer
    model, tokenizer = get_model_tokenizer(inference_args, general_args)

    # Load classifier
    inference_args.model_name_or_path = inference_args.classifier_model_name_or_path
    classifier_model, classifier_tokenizer = get_model_tokenizer(
        inference_args, general_args, model_type='classifier')

    classifier = classify.SponsorBlockClassificationPipeline(
        classifier_model, classifier_tokenizer)

    # Reset to original model_name_or_path
    inference_args.model_name_or_path = original_path

    return model, tokenizer, classifier


def get_model_tokenizer(model_args: ModelArguments, general_args: Union[GeneralArguments, TrainingArguments] = None, config_args=None, model_type='seq2seq'):
    if model_args.model_name_or_path is None:
        raise ModelLoadError('Must specify --model_name_or_path')

    if config_args is None:
        config_args = {}

    use_auth_token = True if model_args.use_auth_token else None

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=use_auth_token,
        **config_args
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=use_auth_token,
    )

    model_type = AutoModelForSeq2SeqLM if model_type == 'seq2seq' else AutoModelForSequenceClassification
    model = model_type.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=use_auth_token,
    )

    # Add custom tokens
    CustomTokens.add_custom_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    # Potentially move model to gpu
    if general_args is not None and not general_args.no_cuda:
        model.to('cuda' if torch.cuda.is_available() else 'cpu')

    return model, tokenizer
