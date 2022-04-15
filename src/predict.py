
from transformers import HfArgumentParser
from dataclasses import dataclass, field
import logging
from shared import CustomTokens, extract_sponsor_matches, GeneralArguments, seconds_to_time
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
from errors import TranscriptError
from model import get_model_tokenizer_classifier, InferenceArguments

logging.basicConfig()
logger = logging.getLogger(__name__)


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


MATCH_WINDOW = 25       # Increase for accuracy, but takes longer: O(n^3)
MERGE_TIME_WITHIN = 8   # Merge predictions if they are within x seconds

# Any prediction whose start time is <= this will be set to start at 0
START_TIME_ZERO_THRESHOLD = 0.08


def filter_and_add_probabilities(predictions, classifier, min_probability):
    """Use classifier to filter predictions"""
    if not predictions:
        return predictions

    # We update the predicted category from the extractive transformer
    # if the classifier is confident enough it is another category

    texts = [
        preprocess.clean_text(' '.join([x['text'] for x in pred['words']]))
        for pred in predictions
    ]
    classifications = classifier(texts)

    filtered_predictions = []
    for prediction, probabilities in zip(predictions, classifications):
        predicted_probabilities = {
            p['label'].lower(): p['score'] for p in probabilities}

        # Get best category + probability
        classifier_category = max(
            predicted_probabilities, key=predicted_probabilities.get)
        classifier_probability = predicted_probabilities[classifier_category]

        if (prediction['category'] not in predicted_probabilities) \
                or (classifier_category != 'none' and classifier_probability > 0.5):  # TODO make param
            # Unknown category or we are confident enough to overrule,
            # so change category to what was predicted by classifier
            prediction['category'] = classifier_category

        if prediction['category'] == 'none':
            continue  # Ignore if categorised as nothing

        prediction['probability'] = predicted_probabilities[prediction['category']]

        if min_probability is not None and prediction['probability'] < min_probability:
            continue  # Ignore if below threshold

        # TODO add probabilities, but remove None and normalise rest
        prediction['probabilities'] = predicted_probabilities

        # if prediction['probability'] < classifier_args.min_probability:
        #     continue

        filtered_predictions.append(prediction)

    return filtered_predictions


def predict(video_id, model, tokenizer, segmentation_args, words=None, classifier=None, min_probability=None):
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

    if classifier is not None:
        predictions = filter_and_add_probabilities(
            predictions, classifier, min_probability)

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


def predict_sponsor_from_texts(texts, model, tokenizer):
    clean_texts = list(map(preprocess.clean_text, texts))
    return predict_sponsor_from_cleaned_texts(clean_texts, model, tokenizer)


def predict_sponsor_from_cleaned_texts(cleaned_texts, model, tokenizer):
    """Given a body of text, predict the words which are part of the sponsor"""
    model_device = next(model.parameters()).device

    decoded_outputs = []
    # Do individually, to avoid running out of memory for long videos
    for cleaned_words in cleaned_texts:
        text = CustomTokens.EXTRACT_SEGMENTS_PREFIX.value + \
            ' '.join(cleaned_words)
        input_ids = tokenizer(text, return_tensors='pt',
                              truncation=True).input_ids.to(model_device)

        # Optimise output length so that we do not generate unnecessarily long texts
        max_out_len = round(min(
            max(
                len(input_ids[0])/SAFETY_TOKENS_PERCENTAGE,
                len(input_ids[0]) + MIN_SAFETY_TOKENS
            ),
            model.model_dim)
        )

        outputs = model.generate(input_ids, max_length=max_out_len)
        decoded_outputs.append(tokenizer.decode(
            outputs[0], skip_special_tokens=True))

    return decoded_outputs


def segments_to_predictions(segments, model, tokenizer):
    predicted_time_ranges = []

    cleaned_texts = [
        [x['cleaned'] for x in cleaned_segment]
        for cleaned_segment in segments
    ]

    sponsorship_texts = predict_sponsor_from_cleaned_texts(
        cleaned_texts, model, tokenizer)

    matches = extract_sponsor_matches(sponsorship_texts)

    for segment_matches, cleaned_batch, segment in zip(matches, cleaned_texts, segments):

        for match in segment_matches:  # one segment might contain multiple sponsors/ir/selfpromos

            matched_text = match['text'].split()

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
        start_time = range['start'] if range['start'] > START_TIME_ZERO_THRESHOLD else 0
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
    logger.setLevel(logging.DEBUG)

    hf_parser = HfArgumentParser((
        PredictArguments,
        SegmentationArguments,
        GeneralArguments
    ))
    predict_args, segmentation_args, general_args = hf_parser.parse_args_into_dataclasses()

    if not predict_args.video_ids:
        logger.error(
            'No video IDs supplied. Use `--video_id`, `--video_ids`, or `--channel_id`.')
        return

    model, tokenizer, classifier = get_model_tokenizer_classifier(
        predict_args, general_args)

    for video_id in predict_args.video_ids:
        try:
            predictions = predict(video_id, model, tokenizer, segmentation_args,
                                  classifier=classifier,
                                  min_probability=predict_args.min_probability)
        except TranscriptError:
            logger.warning(f'No transcript available for {video_id}')
            continue
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        if not predictions:
            logger.info(f'No predictions found for {video_url}')
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
