
from model import get_model_tokenizer_classifier, InferenceArguments
from utils import jaccard, safe_print
from transformers import HfArgumentParser
from preprocess import get_words, clean_text
from shared import GeneralArguments, DatasetArguments
from predict import predict
from segment import extract_segment, word_start, word_end, SegmentationArguments, add_labels_to_words
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import json
import os
import random
from shared import seconds_to_time
from urllib.parse import quote
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)


@dataclass
class EvaluationArguments(InferenceArguments):
    """Arguments pertaining to how evaluation will occur."""
    output_file: Optional[str] = field(
        default='metrics.csv',
        metadata={
            'help': 'Save metrics to output file'
        }
    )

    skip_missing: bool = field(
        default=False,
        metadata={
            'help': 'Whether to skip checking for missing segments. If False, predictions will be made.'
        }
    )
    skip_incorrect: bool = field(
        default=False,
        metadata={
            'help': 'Whether to skip checking for incorrect segments. If False, classifications will be made on existing segments.'
        }
    )


def attach_predictions_to_sponsor_segments(predictions, sponsor_segments):
    """Attach sponsor segments to closest prediction"""
    for prediction in predictions:
        prediction['best_overlap'] = 0
        prediction['best_sponsorship'] = None

        # Assign predictions to actual (labelled) sponsored segments
        for sponsor_segment in sponsor_segments:
            j = jaccard(prediction['start'], prediction['end'],
                        sponsor_segment['start'], sponsor_segment['end'])
            if prediction['best_overlap'] < j:
                prediction['best_overlap'] = j
                prediction['best_sponsorship'] = sponsor_segment

    return sponsor_segments


def calculate_metrics(labelled_words, predictions):

    metrics = {
        'true_positive': 0,  # Is sponsor, predicted sponsor
        # Is sponsor, predicted not sponsor (i.e., missed it - bad)
        'false_negative': 0,
        # Is not sponsor, predicted sponsor (classified incorectly, not that bad since we do manual checking afterwards)
        'false_positive': 0,
        'true_negative': 0,  # Is not sponsor, predicted not sponsor
    }

    metrics['video_duration'] = word_end(
        labelled_words[-1])-word_start(labelled_words[0])

    for index, word in enumerate(labelled_words):
        if index >= len(labelled_words) - 1:
            continue

        duration = word_end(word) - word_start(word)

        predicted_sponsor = False
        for p in predictions:
            # Is in some prediction
            if p['start'] <= word['start'] <= p['end']:
                predicted_sponsor = True
                break

        if predicted_sponsor:
            # total_positive_time += duration
            if word.get('category') is not None:  # Is actual sponsor
                metrics['true_positive'] += duration
            else:
                metrics['false_positive'] += duration
        else:
            # total_negative_time += duration
            if word.get('category') is not None:  # Is actual sponsor
                metrics['false_negative'] += duration
            else:
                metrics['true_negative'] += duration

    # NOTE In cases where we encounter division by 0, we say that the value is 1
    # https://stats.stackexchange.com/a/1775
    # (Precision) TP+FP=0: means that all instances were predicted as negative
    # (Recall)    TP+FN=0: means that there were no positive cases in the input data

    # The fraction of predictions our model got right
    # Can simplify, but use full formula
    z = metrics['true_positive'] + metrics['true_negative'] + \
        metrics['false_positive'] + metrics['false_negative']
    metrics['accuracy'] = (
        (metrics['true_positive'] + metrics['true_negative']) / z) if z > 0 else 1

    # What proportion of positive identifications was actually correct?
    z = metrics['true_positive'] + metrics['false_positive']
    metrics['precision'] = (metrics['true_positive'] / z) if z > 0 else 1

    # What proportion of actual positives was identified correctly?
    z = metrics['true_positive'] + metrics['false_negative']
    metrics['recall'] = (metrics['true_positive'] / z) if z > 0 else 1

    # https://deepai.org/machine-learning-glossary-and-terms/f-score

    s = metrics['precision'] + metrics['recall']
    metrics['f-score'] = (2 * (metrics['precision'] *
                               metrics['recall']) / s) if s > 0 else 0

    return metrics


def main():
    logger.setLevel(logging.DEBUG)

    hf_parser = HfArgumentParser((
        EvaluationArguments,
        DatasetArguments,
        SegmentationArguments,
        GeneralArguments
    ))

    evaluation_args, dataset_args, segmentation_args, general_args = hf_parser.parse_args_into_dataclasses()

    if evaluation_args.skip_missing and evaluation_args.skip_incorrect:
        logger.error('ERROR: Nothing to do')
        return

    # Load labelled data:
    final_path = os.path.join(
        dataset_args.data_dir, dataset_args.processed_file)

    if not os.path.exists(final_path):
        logger.error('ERROR: Processed database not found.\n'
                     f'Run `python src/preprocess.py --update_database --do_create` to generate "{final_path}".')
        return

    model, tokenizer, classifier = get_model_tokenizer_classifier(
        evaluation_args, general_args)

    with open(final_path) as fp:
        final_data = json.load(fp)

    if evaluation_args.video_ids:  # Use specified
        video_ids = evaluation_args.video_ids

    else:  # Use items found in preprocessed database
        video_ids = list(final_data.keys())
        random.shuffle(video_ids)

        if evaluation_args.start_index is not None:
            video_ids = video_ids[evaluation_args.start_index:]

        if evaluation_args.max_videos is not None:
            video_ids = video_ids[:evaluation_args.max_videos]

    out_metrics = []

    all_metrics = {}
    if not evaluation_args.skip_missing:
        all_metrics['total_prediction_accuracy'] = 0
        all_metrics['total_prediction_precision'] = 0
        all_metrics['total_prediction_recall'] = 0
        all_metrics['total_prediction_fscore'] = 0

    if not evaluation_args.skip_incorrect:
        all_metrics['classifier_segment_correct'] = 0
        all_metrics['classifier_segment_count'] = 0

    metric_count = 0

    postfix_info = {}

    try:
        with tqdm(video_ids) as progress:
            for video_index, video_id in enumerate(progress):
                progress.set_description(f'Processing {video_id}')

                words = get_words(video_id)
                if not words:
                    continue

                # Get labels
                sponsor_segments = final_data.get(video_id)

                # Reset previous
                missed_segments = []
                incorrect_segments = []

                current_metrics = {
                    'video_id': video_id
                }
                metric_count += 1

                if not evaluation_args.skip_missing:  # Make predictions
                    predictions = predict(video_id, model, tokenizer, segmentation_args,
                                          classifier=classifier,
                                          min_probability=evaluation_args.min_probability)

                    if sponsor_segments:
                        labelled_words = add_labels_to_words(
                            words, sponsor_segments)

                        current_metrics.update(
                            calculate_metrics(labelled_words, predictions))

                        all_metrics['total_prediction_accuracy'] += current_metrics['accuracy']
                        all_metrics['total_prediction_precision'] += current_metrics['precision']
                        all_metrics['total_prediction_recall'] += current_metrics['recall']
                        all_metrics['total_prediction_fscore'] += current_metrics['f-score']

                        # Just for display purposes
                        postfix_info.update({
                            'accuracy': all_metrics['total_prediction_accuracy']/metric_count,
                            'precision':  all_metrics['total_prediction_precision']/metric_count,
                            'recall':  all_metrics['total_prediction_recall']/metric_count,
                            'f-score': all_metrics['total_prediction_fscore']/metric_count,
                        })

                        sponsor_segments = attach_predictions_to_sponsor_segments(
                            predictions, sponsor_segments)

                        # Identify possible issues:
                        for prediction in predictions:
                            if prediction['best_sponsorship'] is not None:
                                continue

                            prediction_words = prediction.pop('words', [])

                            # Attach original text to missed segments
                            prediction['text'] = ' '.join(
                                x['text'] for x in prediction_words)
                            missed_segments.append(prediction)

                    else:
                        # Not in database (all segments missed)
                        missed_segments = predictions

                if not evaluation_args.skip_incorrect and sponsor_segments:
                    # Check for incorrect segments using the classifier

                    segments_to_check = []
                    texts = []  # Texts to send through tokenizer
                    for sponsor_segment in sponsor_segments:
                        segment_words = extract_segment(
                            words,  sponsor_segment['start'],  sponsor_segment['end'])
                        sponsor_segment['text'] = ' '.join(
                            x['text'] for x in segment_words)

                        duration = sponsor_segment['end'] - \
                            sponsor_segment['start']
                        wps = (len(segment_words) /
                               duration) if duration > 0 else 0
                        if wps < 1.5:
                            continue

                        # Do not worry about those that are locked or have enough votes
                        # or segment['votes'] > 5:
                        if sponsor_segment['locked']:
                            continue

                        sponsor_segment['cleaned_text'] = clean_text(
                            sponsor_segment['text'])
                        texts.append(sponsor_segment['cleaned_text'])
                        segments_to_check.append(sponsor_segment)

                    if segments_to_check:  # Some segments to check

                        segments_scores = classifier(texts)

                        num_correct = 0
                        for segment, scores in zip(segments_to_check, segments_scores):
                            all_metrics['classifier_segment_count'] += 1

                            prediction = max(scores, key=lambda x: x['score'])
                            predicted_category = prediction['label'].lower()

                            if predicted_category == segment['category']:
                                num_correct += 1
                                continue  # Ignore correct segments

                            segment.update({
                                'predicted': predicted_category,
                                'scores': scores
                            })

                            incorrect_segments.append(segment)

                        current_metrics['num_segments'] = len(
                            segments_to_check)
                        current_metrics['classified_correct'] = num_correct

                        all_metrics['classifier_segment_correct'] += num_correct

                    postfix_info['classifier_accuracy'] = all_metrics['classifier_segment_correct'] / \
                        all_metrics['classifier_segment_count']

                out_metrics.append(current_metrics)
                progress.set_postfix(postfix_info)

                if missed_segments or incorrect_segments:

                    if evaluation_args.output_as_json:
                        to_print = {'video_id': video_id}

                        if missed_segments:
                            to_print['missed'] = missed_segments

                        if incorrect_segments:
                            to_print['incorrect'] = incorrect_segments

                        safe_print(json.dumps(to_print))

                    else:
                        safe_print(
                            f'Issues identified for {video_id} (#{video_index})')
                        # Potentially missed segments (model predicted, but not in database)
                        if missed_segments:
                            safe_print(' - Missed segments:')
                            segments_to_submit = []
                            for i, missed_segment in enumerate(missed_segments, start=1):
                                safe_print(f'\t#{i}:', seconds_to_time(
                                    missed_segment['start']), '-->', seconds_to_time(missed_segment['end']))
                                safe_print('\t\tText: "',
                                           missed_segment['text'], '"', sep='')
                                safe_print('\t\tCategory:',
                                           missed_segment.get('category'))
                                if 'probability' in missed_segment:
                                    safe_print('\t\tProbability:',
                                               missed_segment['probability'])

                                segments_to_submit.append({
                                    'segment': [missed_segment['start'], missed_segment['end']],
                                    'category': missed_segment['category'].lower(),
                                    'actionType': 'skip'
                                })

                            json_data = quote(json.dumps(segments_to_submit))
                            safe_print(
                                f'\tSubmit: https://www.youtube.com/watch?v={video_id}#segments={json_data}')

                        # Incorrect segments (in database, but incorrectly classified)
                        if incorrect_segments:
                            safe_print(' - Incorrect segments:')
                            for i, incorrect_segment in enumerate(incorrect_segments, start=1):
                                safe_print(f'\t#{i}:', seconds_to_time(
                                    incorrect_segment['start']), '-->', seconds_to_time(incorrect_segment['end']))

                                safe_print(
                                    '\t\tText: "', incorrect_segment['text'], '"', sep='')
                                safe_print(
                                    '\t\tUUID:', incorrect_segment['uuid'])
                                safe_print(
                                    '\t\tVotes:', incorrect_segment['votes'])
                                safe_print(
                                    '\t\tViews:', incorrect_segment['views'])
                                safe_print('\t\tLocked:',
                                           incorrect_segment['locked'])

                                safe_print('\t\tCurrent Category:',
                                           incorrect_segment['category'])
                                safe_print('\t\tPredicted Category:',
                                           incorrect_segment['predicted'])
                                safe_print('\t\tProbabilities:')
                                for item in incorrect_segment['scores']:
                                    safe_print(
                                        f"\t\t\t{item['label']}: {item['score']}")

                        safe_print()

    except KeyboardInterrupt:
        pass

    df = pd.DataFrame(out_metrics)

    df.to_csv(evaluation_args.output_file)
    logger.info(df.mean())


if __name__ == '__main__':
    main()
