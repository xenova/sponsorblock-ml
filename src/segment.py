import preprocess
from shared import CustomTokens
from dataclasses import dataclass, field


@dataclass
class SegmentationArguments:
    pause_threshold: int = field(default=2.5, metadata={
        'help': 'When the time between words is greater than pause threshold, force into a new segment'})


def get_overlapping_chunks_of_tokens(tokens, size, overlap):
    for i in range(0, len(tokens), size-overlap+1):
        yield tokens[i:i+size]


# Generate up to SAFETY_TOKENS_PERCENTAGE*max_tokens tokens
MIN_SAFETY_TOKENS = 8
SAFETY_TOKENS_PERCENTAGE = 0.9765625
# e.g. 512 -> 500, 768 -> 750


# TODO play around with this?
OVERLAP_TOKEN_PERCENTAGE = 0.5  # 0.25


def add_labels_to_words(words, sponsor_segments):

    for sponsor_segment in sponsor_segments:
        for w in extract_segment(words, sponsor_segment['start'], sponsor_segment['end']):
            w['category'] = sponsor_segment['category']

    return words


def generate_labelled_segments(words, tokenizer, segmentation_args, sponsor_segments):
    segments = generate_segments(words, tokenizer, segmentation_args)

    labelled_segments = list(
        map(lambda x: add_labels_to_words(x, sponsor_segments), segments))

    return labelled_segments


def word_start(word):
    return word['start']


def word_end(word):
    return word.get('end', word['start'])


def generate_segments(words, tokenizer, segmentation_args):
    first_pass_segments = []

    for index, word in enumerate(words):
        # Get length of tokenized word
        word['cleaned'] = preprocess.clean_text(word['text'])
        word['num_tokens'] = len(
            tokenizer(word['cleaned'], add_special_tokens=False, truncation=True).input_ids)

        # Add new segment
        if index == 0 or word_start(words[index]) - word_end(words[index-1]) >= segmentation_args.pause_threshold:
            first_pass_segments.append([word])

        else:  # Add to current segment
            first_pass_segments[-1].append(word)

    max_q_size = round(SAFETY_TOKENS_PERCENTAGE * tokenizer.model_max_length)

    buffer_size = OVERLAP_TOKEN_PERCENTAGE*max_q_size  # tokenizer.model_max_length

    # In second pass, we split those segments if too big
    second_pass_segments = []

    for segment in first_pass_segments:
        current_segment_num_tokens = 0
        current_segment = []

        for word in segment:
            new_seg = current_segment_num_tokens + \
                word['num_tokens'] >= max_q_size
            if new_seg:
                # Adding this token would make it have too many tokens
                # We save this batch and create new
                second_pass_segments.append(current_segment)

            # Add tokens to current segment
            current_segment.append(word)
            current_segment_num_tokens += word['num_tokens']

            if not new_seg:
                continue

            # Just created a new segment, so we remove until we only have buffer_size tokens
            last_index = 0
            while current_segment_num_tokens > buffer_size and current_segment:
                current_segment_num_tokens -= current_segment[last_index]['num_tokens']
                last_index += 1

            current_segment = current_segment[last_index:]

        if current_segment:  # Add remaining segment
            second_pass_segments.append(current_segment)

    # Cleaning up, delete 'num_tokens' from each word
    # for segment in second_pass_segments:
    for word in words:
        word.pop('num_tokens', None)

    return second_pass_segments


def extract_segment(words, start, end, map_function=None):
    """Extracts all words with time in [start, end]"""

    a = binary_search(words, 0, len(words), start, True)
    b = min(binary_search(words, 0, len(words), end, False) + 1, len(words))

    to_transform = map_function is not None and callable(map_function)

    return [
        map_function(words[i]) if to_transform else words[i] for i in range(a, b)
    ]


def binary_search(words, start_index, end_index, time, below):
    """Binary search to get first index of word whose start/end time is greater/less than some value"""
    if start_index >= end_index:
        return end_index

    middle_index = (start_index + end_index) // 2

    middle_time = word_start(
        words[middle_index]) if below else word_end(words[middle_index])

    # TODO if above: if time < middle_time binary_search(start, middle-1)
    if time <= middle_time:
        return binary_search(words, start_index, middle_index, time, below)
    else:
        return binary_search(words, middle_index + 1, end_index, time, below)
