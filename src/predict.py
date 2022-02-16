import itertools
import base64
import re
import requests
import json
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from dataclasses import dataclass, field
import logging
import os
import itertools
from utils import re_findall
from shared import CustomTokens, START_SEGMENT_TEMPLATE, END_SEGMENT_TEMPLATE, OutputArguments, seconds_to_time
from typing import Optional
from segment import (
    generate_segments,
    extract_segment,
    MIN_SAFETY_TOKENS,
    SAFETY_TOKENS_PERCENTAGE,
    word_start,
    word_end,
    SegmentationArguments
)
import preprocess
from errors import PredictionException, TranscriptError, ModelLoadError, ClassifierLoadError
from model import ModelArguments, get_classifier_vectorizer, get_model_tokenizer

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
class InferenceArguments:

    model_path: str = field(
        default='Xenova/sponsorblock-small',
        metadata={
            'help': 'Path to pretrained model used for prediction'
        }
    )
    cache_dir: Optional[str] = ModelArguments.__dataclass_fields__['cache_dir']

    output_dir: Optional[str] = OutputArguments.__dataclass_fields__[
        'output_dir']

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

    no_cuda: bool = ModelArguments.__dataclass_fields__['no_cuda']

    def __post_init__(self):
        # Try to load model from latest checkpoint
        if self.model_path is None:
            if os.path.exists(self.output_dir):
                last_checkpoint = get_last_checkpoint(self.output_dir)
                if last_checkpoint is not None:
                    self.model_path = last_checkpoint
                else:
                    raise ModelLoadError(
                        'Unable to load model from checkpoint, explicitly set `--model_path`')
            else:
                raise ModelLoadError(
                    f'Unable to find model in {self.output_dir}, explicitly set `--model_path`')

        if any(len(video_id) != 11 for video_id in self.video_ids):
            raise PredictionException('Invalid video IDs (length not 11)')

        if self.channel_id is not None:
            start = self.start_index or 0
            end = None if self.max_videos is None else start + self.max_videos

            channel_video_ids = list(itertools.islice(get_all_channel_vids(
                self.channel_id), start, end))
            logger.info(
                f'Found {len(channel_video_ids)} for channel {self.channel_id}')

            self.video_ids += channel_video_ids


@dataclass
class PredictArguments(InferenceArguments):
    video_id: str = field(
        default=None,
        metadata={
            'help': 'Video to predict segments for'}
    )

    def __post_init__(self):
        if self.video_id is not None:
            self.video_ids.append(self.video_id)

        super().__post_init__()


_SEGMENT_START = START_SEGMENT_TEMPLATE.format(r'(?P<category>\w+)')
_SEGMENT_END = END_SEGMENT_TEMPLATE.format(r'\w+')
SEGMENT_MATCH_RE = fr'{_SEGMENT_START}\s*(?P<text>.*?)\s*(?:{_SEGMENT_END}|$)'

MATCH_WINDOW = 25       # Increase for accuracy, but takes longer: O(n^3)
MERGE_TIME_WITHIN = 8   # Merge predictions if they are within x seconds


@dataclass(frozen=True, eq=True)
class ClassifierArguments:
    classifier_model: Optional[str] = field(
        default='Xenova/sponsorblock-classifier',
        metadata={
            'help': 'Use a pretrained classifier'
        }
    )

    classifier_dir: Optional[str] = field(
        default='classifiers',
        metadata={
            'help': 'The directory that contains the classifier and vectorizer.'
        }
    )

    classifier_file: Optional[str] = field(
        default='classifier.pickle',
        metadata={
            'help': 'The name of the classifier'
        }
    )

    vectorizer_file: Optional[str] = field(
        default='vectorizer.pickle',
        metadata={
            'help': 'The name of the vectorizer'
        }
    )

    min_probability: float = field(
        default=0.5, metadata={'help': 'Remove all predictions whose classification probability is below this threshold.'})


def filter_and_add_probabilities(predictions, classifier_args):
    """Use classifier to filter predictions"""
    if not predictions:
        return predictions

    classifier, vectorizer = get_classifier_vectorizer(classifier_args)

    transformed_segments = vectorizer.transform([
        preprocess.clean_text(' '.join([x['text'] for x in pred['words']]))
        for pred in predictions
    ])
    probabilities = classifier.predict_proba(transformed_segments)

    # Transformer sometimes says segment is of another category, so we
    # update category and probabilities if classifier is confident it is another category
    filtered_predictions = []
    for prediction, probabilities in zip(predictions, probabilities):
        predicted_probabilities = {k: v for k,
                                   v in zip(CATEGORIES, probabilities)}

        # Get best category + probability
        classifier_category = max(
            predicted_probabilities, key=predicted_probabilities.get)
        classifier_probability = predicted_probabilities[classifier_category]

        if classifier_category is None and classifier_probability > classifier_args.min_probability:
            continue  # Ignore

        if (prediction['category'] not in predicted_probabilities) \
                or (classifier_category is not None and classifier_probability > 0.5):  # TODO make param
            # Unknown category or we are confident enough to overrule,
            # so change category to what was predicted by classifier
            prediction['category'] = classifier_category

        prediction['probability'] = predicted_probabilities[prediction['category']]

        # TODO add probabilities, but remove None and normalise rest
        prediction['probabilities'] = predicted_probabilities

        # if prediction['probability'] < classifier_args.min_probability:
        #     continue

        filtered_predictions.append(prediction)

    return filtered_predictions


def predict(video_id, model, tokenizer, segmentation_args, words=None, classifier_args=None):
    # Allow words to be passed in so that we don't have to get the words if we already have them
    if words is None:
        words = preprocess.get_words(video_id)
        if not words:
            raise TranscriptError('Unable to retrieve transcript')

    segments = generate_segments(
        words,
        tokenizer,
        segmentation_args
    )

    predictions = segments_to_predictions(segments, model, tokenizer)
    # Add words back to time_ranges
    for prediction in predictions:
        # Stores words in the range
        prediction['words'] = extract_segment(
            words, prediction['start'], prediction['end'])

    # TODO add back
    if classifier_args is not None:
        try:
            predictions = filter_and_add_probabilities(
                predictions, classifier_args)
        except ClassifierLoadError:
            print('Unable to load classifer')

    return predictions


def greedy_match(list, sublist):
    # Return index and length of longest matching sublist

    best_i = -1
    best_j = -1
    best_k = 0

    for i in range(len(list)):  # Start position in main list
        for j in range(len(sublist)):  # Start position in sublist
            for k in range(len(sublist)-j, 0, -1):  # Width of sublist window
                if k > best_k and list[i:i+k] == sublist[j:j+k]:
                    best_i, best_j, best_k = i, j, k
                    break  # Since window size decreases

    return best_i, best_j, best_k


CATEGORIES = [None, 'SPONSOR', 'SELFPROMO', 'INTERACTION']


def predict_sponsor_text(text, model, tokenizer):
    """Given a body of text, predict the words which are part of the sponsor"""
    model_device = next(model.parameters()).device
    input_ids = tokenizer(
        f'{CustomTokens.EXTRACT_SEGMENTS_PREFIX.value} {text}', return_tensors='pt', truncation=True).input_ids.to(model_device)

    max_out_len = round(min(
        max(
            len(input_ids[0])/SAFETY_TOKENS_PERCENTAGE,
            len(input_ids[0]) + MIN_SAFETY_TOKENS
        ),
        model.model_dim))
    outputs = model.generate(input_ids, max_length=max_out_len)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def predict_sponsor_matches(text, model, tokenizer):
    sponsorship_text = predict_sponsor_text(text, model, tokenizer)

    if CustomTokens.NO_SEGMENT.value in sponsorship_text:
        return []

    return re_findall(SEGMENT_MATCH_RE, sponsorship_text)


def segments_to_predictions(segments, model, tokenizer):
    predicted_time_ranges = []

    # TODO pass to model simultaneously, not in for loop
    # use 2d array for input ids
    for segment in segments:
        cleaned_batch = [preprocess.clean_text(
            word['text']) for word in segment]
        batch_text = ' '.join(cleaned_batch)

        matches = predict_sponsor_matches(batch_text, model, tokenizer)

        for match in matches:
            matched_text = match['text'].split()
            # TODO skip if too short

            i1, j1, k1 = greedy_match(
                cleaned_batch, matched_text[:MATCH_WINDOW])
            i2, j2, k2 = greedy_match(
                cleaned_batch, matched_text[-MATCH_WINDOW:])

            extracted_words = segment[i1:i2+k2]

            if not extracted_words:
                continue

            predicted_time_ranges.append({
                'start': word_start(extracted_words[0]),
                'end': word_end(extracted_words[-1]),
                'category': match['category']
            })

    # Necessary to sort matches by start time
    predicted_time_ranges.sort(key=word_start)

    # Merge overlapping predictions and sponsorships that are close together
    # Caused by model having max input size

    prev_prediction = None

    final_predicted_time_ranges = []
    for range in predicted_time_ranges:
        start_time = range['start']
        end_time = range['end']

        if prev_prediction is not None and \
                (start_time <= prev_prediction['end'] <= end_time or    # Merge overlapping segments
                    (range['category'] == prev_prediction['category']   # Merge disconnected segments if same category and within threshold
                        and start_time - prev_prediction['end'] <= MERGE_TIME_WITHIN)):
            # Extend last prediction range
            final_predicted_time_ranges[-1]['end'] = end_time

        else:  # No overlap, is a new prediction
            final_predicted_time_ranges.append({
                'start': start_time,
                'end': end_time,
                'category': range['category']
            })

        prev_prediction = range

    return final_predicted_time_ranges


def main():
    # Test on unseen data
    # logging.getLogger().setLevel(logging.DEBUG)

    hf_parser = HfArgumentParser((
        PredictArguments,
        SegmentationArguments,
        ClassifierArguments
    ))
    predict_args, segmentation_args, classifier_args = hf_parser.parse_args_into_dataclasses()

    if not predict_args.video_ids:
        logger.error(
            'No video IDs supplied. Use `--video_id`, `--video_ids`, or `--channel_id`.')
        return

    model, tokenizer = get_model_tokenizer(
        predict_args.model_path, predict_args.cache_dir, predict_args.no_cuda)

    for video_id in predict_args.video_ids:
        video_id = video_id.strip()
        try:
            predictions = predict(video_id, model, tokenizer,
                                  segmentation_args, classifier_args=classifier_args)
        except TranscriptError:
            logger.warning('No transcript available for', video_id, end='\n\n')
            continue
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        if not predictions:
            logger.info('No predictions found for', video_url, end='\n\n')
            continue

        # TODO use predict_args.output_as_json
        print(len(predictions), 'predictions found for', video_url)
        for index, prediction in enumerate(predictions, start=1):
            print(f'Prediction #{index}:')
            print('Text: "',
                  ' '.join([w['text'] for w in prediction['words']]), '"', sep='')
            print('Time:', seconds_to_time(
                prediction['start']), '\u2192', seconds_to_time(prediction['end']))
            print('Category:', prediction.get('category'))
            if 'probability' in prediction:
                print('Probability:', prediction['probability'])
            print()
        print()


if __name__ == '__main__':
    main()
