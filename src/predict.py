from transformers.trainer_utils import get_last_checkpoint
from shared import OutputArguments
from typing import Optional
from segment import (
    generate_segments,
    extract_segment,
    SAFETY_TOKENS,
    CustomTokens,
    word_start,
    word_end,
    SegmentationArguments
)
import preprocess
import re
from errors import TranscriptError
from model import get_classifier_vectorizer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from shared import device
import logging


def seconds_to_time(seconds):
    h, remainder = divmod(abs(int(seconds)), 3600)
    m, s = divmod(remainder, 60)
    return f"{'-' if seconds < 0 else ''}{h:02}:{m:02}:{s:02}"


@dataclass
class TrainingOutputArguments:

    model_path: str = field(
        default=None,
        metadata={
            'help': 'Path to pretrained model used for prediction'}
    )

    output_dir: Optional[str] = OutputArguments.__dataclass_fields__[
        'output_dir']

    def __post_init__(self):
        if self.model_path is not None:
            return

        last_checkpoint = get_last_checkpoint(self.output_dir)
        if last_checkpoint is not None:
            self.model_path = last_checkpoint
        else:
            raise Exception(
                'Unable to find model, explicitly set `--model_path`')


@dataclass
class PredictArguments(TrainingOutputArguments):
    video_id: str = field(
        default=None,
        metadata={
            'help': 'Video to predict sponsorship segments for'}
    )


SPONSOR_MATCH_RE = fr'(?<={CustomTokens.START_SPONSOR.value})\s*(.*?)\s*(?={CustomTokens.END_SPONSOR.value}|$)'

MATCH_WINDOW = 25       # Increase for accuracy, but takes longer: O(n^3)
MERGE_TIME_WITHIN = 8   # Merge predictions if they are within x seconds


@dataclass
class ClassifierArguments:
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


def filter_predictions(predictions, classifier, vectorizer, classifier_args):
    """Use classifier to filter predictions"""
    if not predictions:
        return predictions

    transformed_segments = vectorizer.transform([
        preprocess.clean_text(' '.join([x['text'] for x in pred['words']]))
        for pred in predictions
    ])
    probabilities = classifier.predict_proba(transformed_segments)

    filtered_predictions = []
    for prediction, probability in zip(predictions, probabilities):
        prediction['probability'] = probability[1]

        if prediction['probability'] >= classifier_args.min_probability:
            filtered_predictions.append(prediction)
        # else:
            # print('removing segment', prediction)

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

    predictions = segments_to_prediction_times(segments, model, tokenizer)

    # Add words back to time_ranges
    for prediction in predictions:
        # Stores words in the range
        prediction['words'] = extract_segment(
            words, prediction['start'], prediction['end'])

    if classifier_args is not None:
        classifier, vectorizer = get_classifier_vectorizer(classifier_args)
        predictions = filter_predictions(
            predictions, classifier, vectorizer, classifier_args)

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


DEFAULT_TOKEN_PREFIX = 'summarize: '


def predict_sponsor_text(text, model, tokenizer):
    """Given a body of text, predict the words which are part of the sponsor"""
    input_ids = tokenizer(
        f'{DEFAULT_TOKEN_PREFIX}{text}', return_tensors='pt', truncation=True).input_ids.to(device())

    # Can't be longer than input length + SAFETY_TOKENS or model input dim
    max_out_len = min(len(input_ids[0]) + SAFETY_TOKENS, model.model_dim)
    outputs = model.generate(input_ids, max_length=max_out_len)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def predict_sponsor_matches(text, model, tokenizer):
    sponsorship_text = predict_sponsor_text(text, model, tokenizer)
    if CustomTokens.NO_SPONSOR.value in sponsorship_text:
        return []

    return re.findall(SPONSOR_MATCH_RE, sponsorship_text)


def segments_to_prediction_times(segments, model, tokenizer):
    predicted_time_ranges = []

    # TODO pass to model simultaneously, not in for loop
    # use 2d array for input ids
    for segment in segments:
        cleaned_batch = [preprocess.clean_text(
            word['text']) for word in segment]
        batch_text = ' '.join(cleaned_batch)

        matches = predict_sponsor_matches(batch_text, model, tokenizer)

        for match in matches:
            matched_text = match.split()
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
                'end': word_end(extracted_words[-1])
            })

    # Necessary to sort matches by start time
    predicted_time_ranges.sort(key=word_start)

    # Merge overlapping predictions and sponsorships that are close together
    # Caused by model having max input size
    last_end_time = -1
    final_predicted_time_ranges = []
    for range in predicted_time_ranges:
        start_time = range['start']
        end_time = range['end']

        if (start_time <= last_end_time <= end_time) or (last_end_time != -1 and start_time - last_end_time <= MERGE_TIME_WITHIN):
            # Ending time of last segment is in this segment, so we extend last prediction range
            final_predicted_time_ranges[-1]['end'] = end_time

        else:  # No overlap, is a new prediction
            final_predicted_time_ranges.append({
                'start': start_time,
                'end': end_time,
            })

        last_end_time = end_time

    return final_predicted_time_ranges


def main():
    # Test on unseen data
    logging.getLogger().setLevel(logging.DEBUG)

    hf_parser = HfArgumentParser((
        PredictArguments,
        SegmentationArguments,
        ClassifierArguments
    ))
    predict_args, segmentation_args, classifier_args = hf_parser.parse_args_into_dataclasses()

    if predict_args.video_id is None:
        print('No video ID supplied. Use `--video_id`.')
        return

    model = AutoModelForSeq2SeqLM.from_pretrained(predict_args.model_path)
    model.to(device())

    tokenizer = AutoTokenizer.from_pretrained(predict_args.model_path)

    predict_args.video_id = predict_args.video_id.strip()
    print(
        f'Predicting for https://www.youtube.com/watch?v={predict_args.video_id}')
    predictions = predict(predict_args.video_id, model, tokenizer,
                          segmentation_args, classifier_args=classifier_args)

    for prediction in predictions:
        print(' '.join([w['text'] for w in prediction['words']]))
        print(seconds_to_time(prediction['start']),
              '-->', seconds_to_time(prediction['end']))
        print(prediction['start'], '-->', prediction['end'])
        print(prediction['probability'])
        print()

    print()


if __name__ == '__main__':
    main()
