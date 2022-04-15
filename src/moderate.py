import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser
)

from train_classifier import ClassifierModelArguments
from shared import CATEGORIES, DatasetArguments
from tqdm import tqdm

from preprocess import get_words, clean_text
from segment import extract_segment
import os
import json
import numpy as np


def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ClassifierModelArguments, DatasetArguments))
    model_args, dataset_args = parser.parse_args_into_dataclasses()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    processed_db_path = os.path.join(
        dataset_args.data_dir, dataset_args.processed_database)
    with open(processed_db_path) as fp:
        data = json.load(fp)

    mapped_categories = {
        str(v).lower(): k for k, v in enumerate(CATEGORIES)
    }

    for video_id, segments in tqdm(data.items()):

        words = get_words(video_id)

        if not words:
            continue  # No/empty transcript for video_id

        valid_segments = []
        texts = []
        for segment in segments:
            segment_words = extract_segment(
                words, segment['start'], segment['end'])
            text = clean_text(' '.join(x['text'] for x in segment_words))

            duration = segment['end'] - segment['start']
            wps = len(segment_words)/duration if duration > 0 else 0
            if wps < 1.5:
                continue

            # Do not worry about those that are locked or have enough votes
            if segment['locked']:  # or segment['votes'] > 5:
                continue

            texts.append(text)
            valid_segments.append(segment)

        if not texts:
            continue  # No valid segments

        model_inputs = tokenizer(
            texts, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            model_outputs = model(**model_inputs)
            outputs = list(map(lambda x: x.numpy(), model_outputs['logits']))

        scores = softmax(outputs)

        for segment, text, score in zip(valid_segments, texts, scores):
            predicted_index = score.argmax().item()

            if predicted_index == mapped_categories[segment['category']]:
                continue  # Ignore correct segments

            a = {k: round(float(score[i]), 3)
                 for i, k in enumerate(CATEGORIES)}

            del segment['submission_time']
            segment.update({
                'predicted': str(CATEGORIES[predicted_index]).lower(),
                'text': text,
                'scores': a
            })

            print(json.dumps(segment))


if __name__ == "__main__":
    main()
