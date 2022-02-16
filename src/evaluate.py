
from model import get_model_tokenizer
from utils import jaccard
from transformers import HfArgumentParser
from preprocess import DatasetArguments, get_words
from shared import GeneralArguments
from predict import ClassifierArguments, predict, InferenceArguments
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


def attach_predictions_to_sponsor_segments(predictions, sponsor_segments):
    """Attach sponsor segments to closest prediction"""
    for prediction in predictions:
        prediction['best_overlap'] = 0
        prediction['best_sponsorship'] = None

    # Assign predictions to actual (labelled) sponsored segments
    for sponsor_segment in sponsor_segments:
        sponsor_segment['best_overlap'] = 0
        sponsor_segment['best_prediction'] = None

        for prediction in predictions:

            j = jaccard(prediction['start'], prediction['end'],
                        sponsor_segment['start'], sponsor_segment['end'])
            if sponsor_segment['best_overlap'] < j:
                sponsor_segment['best_overlap'] = j
                sponsor_segment['best_prediction'] = prediction

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

        # TODO make sure words with manual transcripts
        duration = labelled_words[index+1]['start'] - word['start']

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
    hf_parser = HfArgumentParser((
        EvaluationArguments,
        DatasetArguments,
        SegmentationArguments,
        ClassifierArguments,
        GeneralArguments
    ))

    evaluation_args, dataset_args, segmentation_args, classifier_args, _ = hf_parser.parse_args_into_dataclasses()

    # Load labelled data:
    final_path = os.path.join(
        dataset_args.data_dir, dataset_args.processed_file)

    if not os.path.exists(final_path):
        logger.error('ERROR: Processed database not found.\n'
                     f'Run `python src/preprocess.py --update_database --do_create` to generate "{final_path}".')
        return

    model, tokenizer = get_model_tokenizer(
        evaluation_args.model_path, evaluation_args.cache_dir, evaluation_args.no_cuda)

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

    # TODO option to choose categories

    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_fscore = 0

    out_metrics = []

    try:
        with tqdm(video_ids) as progress:
            for video_index, video_id in enumerate(progress):
                progress.set_description(f'Processing {video_id}')

                words = get_words(video_id)
                if not words:
                    continue

                # Make predictions
                predictions = predict(video_id, model, tokenizer,
                                      segmentation_args, words, classifier_args)

                # Get labels
                sponsor_segments = final_data.get(video_id)
                if sponsor_segments:
                    labelled_words = add_labels_to_words(
                        words, sponsor_segments)
                    met = calculate_metrics(labelled_words, predictions)
                    met['video_id'] = video_id

                    out_metrics.append(met)

                    total_accuracy += met['accuracy']
                    total_precision += met['precision']
                    total_recall += met['recall']
                    total_fscore += met['f-score']

                    progress.set_postfix({
                        'accuracy': total_accuracy/len(out_metrics),
                        'precision':  total_precision/len(out_metrics),
                        'recall':  total_recall/len(out_metrics),
                        'f-score': total_fscore/len(out_metrics)
                    })

                    labelled_predicted_segments = attach_predictions_to_sponsor_segments(
                        predictions, sponsor_segments)

                    # Identify possible issues:
                    missed_segments = [
                        prediction for prediction in predictions if prediction['best_sponsorship'] is None]
                    incorrect_segments = [
                        seg for seg in labelled_predicted_segments if seg['best_prediction'] is None]

                    # Add words to incorrect segments
                    for seg in incorrect_segments:
                        seg['words'] = extract_segment(
                            words, seg['start'], seg['end'])

                else:
                    # logger.warning(f'No labels found for {video_id}')
                    # Not in database (all segments missed)
                    missed_segments = predictions
                    incorrect_segments = []

                if missed_segments or incorrect_segments:
                    if evaluation_args.output_as_json:
                        to_print = {'video_id': video_id}

                        for z in missed_segments + incorrect_segments:
                            z['text'] = ' '.join(x['text']
                                                 for x in z.pop('words', []))

                        if missed_segments:
                            to_print['missed'] = missed_segments

                        if incorrect_segments:
                            to_print['incorrect'] = incorrect_segments

                        print(json.dumps(to_print))
                    else:
                        print(
                            f'Issues identified for {video_id} (#{video_index})')
                        # Potentially missed segments (model predicted, but not in database)
                        if missed_segments:
                            print(' - Missed segments:')
                            segments_to_submit = []
                            for i, missed_segment in enumerate(missed_segments, start=1):
                                print(f'\t#{i}:', seconds_to_time(
                                    missed_segment['start']), '-->', seconds_to_time(missed_segment['end']))
                                print('\t\tText: "', ' '.join(
                                    [w['text'] for w in missed_segment['words']]), '"', sep='')
                                print('\t\tCategory:',
                                      missed_segment.get('category'))
                                if 'probability' in missed_segment:
                                    print('\t\tProbability:',
                                          missed_segment['probability'])

                                segments_to_submit.append({
                                    'segment': [missed_segment['start'], missed_segment['end']],
                                    'category': missed_segment['category'].lower(),
                                    'actionType': 'skip'
                                })

                            json_data = quote(json.dumps(segments_to_submit))
                            print(
                                f'\tSubmit: https://www.youtube.com/watch?v={video_id}#segments={json_data}')

                        # Potentially incorrect segments (model didn't predict, but in database)
                        if incorrect_segments:
                            print(' - Incorrect segments:')
                            for i, incorrect_segment in enumerate(incorrect_segments, start=1):
                                print(f'\t#{i}:', seconds_to_time(
                                    incorrect_segment['start']), '-->', seconds_to_time(incorrect_segment['end']))

                                seg_words = extract_segment(
                                    words, incorrect_segment['start'], incorrect_segment['end'])
                                print('\t\tText: "', ' '.join(
                                    [w['text'] for w in seg_words]), '"', sep='')
                                print('\t\tUUID:', incorrect_segment['uuid'])
                                print('\t\tCategory:',
                                      incorrect_segment['category'])
                                print('\t\tVotes:', incorrect_segment['votes'])
                                print('\t\tViews:', incorrect_segment['views'])
                                print('\t\tLocked:',
                                      incorrect_segment['locked'])
                        print()

    except KeyboardInterrupt:
        pass

    df = pd.DataFrame(out_metrics)

    df.to_csv(evaluation_args.output_file)
    logger.info(df.mean())


if __name__ == '__main__':
    main()
