from utils import jaccard, Task, InterruptibleTaskPool
from functools import lru_cache
from datetime import datetime
import itertools
from typing import Optional, List
from datasets import load_dataset
from model import ModelArguments
import segment
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from shared import CATGEGORY_OPTIONS, START_SEGMENT_TEMPLATE, END_SEGMENT_TEMPLATE, GeneralArguments, CustomTokens
import csv
import re
import random
import logging
from youtube_transcript_api import YouTubeTranscriptApi, CouldNotRetrieveTranscript, YouTubeRequestFailed, TooManyRequests
import os
import json
import time
import requests


def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def wordify(transcript, maximum_wps=1):
    """Try to replicate format for automatically generated transcripts"""

    # Do not allow segments to be on screen for too long using maximum_wps
    words = []

    for line_index, line in enumerate(transcript):
        text = line['text'].replace('\n', ' ').strip()
        if not text:
            continue

        start = line['start']
        next_start = transcript[line_index + 1]['start'] \
            if line_index < len(transcript) - 1 else float('inf')

        # Use maximum wps to calculate latest end (to avoid segments which stay on screen too long)
        longest_duration = maximum_wps * text.count(' ')
        latest_end = start + longest_duration
        end = min(start + line['duration'], next_start, latest_end)

        duration = end - start

        indices = find(text, ' ') + [len(text)]
        start_index = 0
        for i in range(len(indices)):
            word = text[start_index:indices[i]].strip()
            if not word:
                continue  # Skip empty words (e.g., \n)
            percentage = start_index/indices[-1]

            w_duration = len(word)/indices[-1] * duration

            w_start = start + percentage * duration

            words.append({
                'start': round(w_start, 3),
                'duration': round(w_duration, 3),
                'end': round(w_start + w_duration, 3),
                'text': word,
            })

            start_index = indices[i] + 1

    return words


def get_manual_words(transcript_list):
    transcript = transcript_list.find_manually_created_transcript(
        ['en-GB', 'en-US', 'en']).fetch()
    return wordify(transcript)


PROFANITY_RAW = '[ __ ]'  # How YouTube transcribes profanity
PROFANITY_CONVERTED = '*****'  # Safer version for tokenizing


# TODO add end time for words
def get_auto_words(transcript_list):
    words = []
    transcript = transcript_list.find_generated_transcript(['en'])
    url = transcript._url + '&fmt=json3'
    info = transcript._http_client.get(url)

    for event in info.json()['events']:
        start_ms = event.get('tStartMs', 0)

        for word in event.get('segs') or []:
            offset_ms = word.get('tOffsetMs', 0)

            texts = word['utf8'].replace(
                PROFANITY_RAW, PROFANITY_CONVERTED
            ).strip().split()

            for text in texts:
                words.append({
                    'start': (start_ms + offset_ms)/1000,
                    'text': text
                })

    return words


def list_transcripts(video_id):
    return YouTubeTranscriptApi.list_transcripts(video_id)


WORDS_TO_REMOVE = [
    CustomTokens.MUSIC.value,
    CustomTokens.APPLAUSE.value,
    CustomTokens.LAUGHTER.value
]


@lru_cache(maxsize=16)
def get_words(video_id, process=True, transcript_type='auto', fallback='manual', filter_words_to_remove=True):
    """Get parsed video transcript with caching system
    returns None if not processed yet and process is False
    """
    transcript_path = os.path.join(  # TODO use relative path to this
        'transcripts', transcript_type, f'{video_id}.json')

    words = None
    try:
        if os.path.exists(transcript_path):  # Load from file
            with open(transcript_path) as fp:
                words = json.load(fp)  # May be empty

        elif process:
            transcript_list = list_transcripts(video_id)

            if transcript_type == 'manual':
                words = get_manual_words(transcript_list)
            else:
                words = get_auto_words(transcript_list)

    except (TooManyRequests, YouTubeRequestFailed):
        raise  # Cannot recover from these errors and do not mark as empty transcript

    except requests.exceptions.ConnectionError:  # Can recover
        time.sleep(10)  # Timeout
        return get_words(video_id, process, transcript_type, fallback)

    except CouldNotRetrieveTranscript:  # Retrying won't solve
        pass  # Mark as empty transcript

    except json.decoder.JSONDecodeError:
        print('JSONDecodeError for', video_id)
        os.remove(transcript_path)  # Remove file and try again
        return get_words(video_id, process, transcript_type, fallback)

    # Tried to process it, but it was empty...
    if process and not os.path.exists(transcript_path):
        with open(transcript_path, 'w') as fp:
            json.dump(words, fp)

    if not words and fallback is not None:
        return get_words(video_id, process, transcript_type=fallback, fallback=None)

    if words and filter_words_to_remove:
        words = list(filter(lambda x: x['text'] not in WORDS_TO_REMOVE, words))

    return words


# TODO make min_sponsor_segment_length param
# TODO rename to extract_segments
def extract_sponsors(words, min_sponsor_segment_length=3):
    if not words or len(words) < min_sponsor_segment_length:
        return []

    paragraphs = []
    current = []
    prev_category = None

    for i in range(len(words) + 1):
        unimportant = i == len(words) or words[i].get('category') is None

        if unimportant or words[i].get('category') != prev_category:
            if current:  # Save the current batch
                paragraphs.append({
                    'words': current,
                    'category': current[-1].get('category'),
                })

                current = []

        if not unimportant:  # Some useful information to save
            current.append(words[i])
            prev_category = words[i].get('category')

    # Remove all too short:
    return list(filter(lambda x: len(x['words']) >= min_sponsor_segment_length, paragraphs))


def clean_text(text):

    # Replace impossibly long words with a special token
    # Usually the result of incorrect labelling
    text = re.sub(r'\w{64,}', CustomTokens.LONG_WORD.value, text)

    SHORT_HYPHENATED_REGEX = r'\w{1,2}(?:-\w{1,2}){3,}(?:-?\w*)'

    # Replace hyphenated URLs with special token
    # For some reason, youtube sometimes transcribes urls in this form:
    # 'b-a-b-b-e-l-dot-com', 'g-e-t-r-o-m-a-n-com'
    # not 'e-commerce'
    text = re.sub(f'{SHORT_HYPHENATED_REGEX}(?:com|org|net)',
                  CustomTokens.HYPHENATED_URL.value, text)

    # Replace short+hyphenated text with a special token. Of the form:
    # 'i-i-i-i-i-i-i-i-i-i-i-i', 'b-u-m-f-u-z-z-l-e', 'v-e-r-i-t-a-s-i-u-m', 'do-do-do-do-do'
    text = re.sub(SHORT_HYPHENATED_REGEX,
                  CustomTokens.SHORT_HYPHENATED.value, text)

    # Replace URLs with URL_TOKEN
    URL_REGEX = r'(?:(?:http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.(?:[a-zA-Z]){2,6}(?:[a-zA-Z0-9\.\&\/\?\:@\-_=#%])*'
    text = re.sub(URL_REGEX, CustomTokens.URL.value, text)

    NUM_REGEX = r'(?:\d+,)*(?:\d*[.])?\d+'

    # Encode specific numeric words
    # Of the form: 12%, 12.34%
    # Usually included in sponsorships
    text = re.sub(f'{NUM_REGEX}%',
                  CustomTokens.NUMBER_PERCENTAGE.value, text)

    # Normal numbers, should not have an effect on sponsorship
    text = re.sub(NUM_REGEX, CustomTokens.NUMBER.value, text)

    # Replace profanity with special token
    text = text.replace(PROFANITY_RAW, CustomTokens.PROFANITY.value)
    text = text.replace(PROFANITY_CONVERTED, CustomTokens.PROFANITY.value)

    return text.strip()


def remove_duplicate_segments(segments):
    # Algorithm based on SponsorBlock algorithm
    # https://blog.ajay.app/voting-and-pseudo-randomness-or-sponsorblock-or-youtube-sponsorship-segment-blocker
    # Find sponsors that are overlapping

    best = []
    for i in segments:
        similar_segments = []
        for j in segments:
            if jaccard(i['start'], i['end'], j['start'], j['end']) > 0.1:  # Some overlap
                similar_segments.append(j)

        if similar_segments:
            best_similar_seg = max(similar_segments, key=lambda item: (
                item['locked'],
                item['votes'],
                item['views'],
                item['reputation']
            ))
            if best_similar_seg not in best:
                best.append(best_similar_seg)

    if len(segments) != len(best):  # Saw some reduction... try again
        return remove_duplicate_segments(best)

    return best


@dataclass
class PreprocessArguments:
    """
    Arguments pertaining to what data we are going to preprocess.
    """
    update_database: bool = field(
        default=False, metadata={'help': 'Download the raw database.'}
    )

    do_create: bool = field(
        default=False, metadata={'help': 'Merge sponsor segments into single file'}
    )

    min_votes: int = field(
        default=0, metadata={'help': 'Minimum number of votes'})
    # Downvotes will make this negative.
    # 1 = At least one positive vote

    min_views: int = field(
        default=5, metadata={'help': 'Minimum number of views a segment must have to be considered. 0 = show all'})

    # min_reputation: int = field(
    #     default=0, metadata={'help': 'Minimum reputation a user must have for the segment to be included'})

    min_date: str = field(
        # default='08/06/2020', # release of v2.0 (https://github.com/ajayyy/SponsorBlock/releases/tag/2.0)
        # release of v3.0 (https://github.com/ajayyy/SponsorBlock/releases/tag/3.0)
        default='20/08/2021',
        # default='01/10/2020', # No more autovote
        metadata={'help': 'Only use submissions from after this date (inclusive)'})

    max_date: str = field(
        # default='01/01/9999', # Include all
        default='27/01/2022',
        metadata={'help': 'Only use videos that have some segment from before this date (exclusive). This allows for videos to have segments be corrected, but ignores new videos (posted after this date) to enter the pool.'})

    do_transcribe: bool = field(
        default=False, metadata={'help': 'Get transcripts for videos'}
    )
    num_jobs: int = field(
        default=4, metadata={'help': 'Number of transcripts to download in parallel'})

    overwrite: bool = field(
        default=False, metadata={'help': 'Overwrite training, testing and validation data, if present.'}
    )

    do_generate: bool = field(
        default=False, metadata={'help': 'Generate labelled data.'}
    )

    do_split: bool = field(
        default=False, metadata={'help': 'Generate training, testing and validation data.'}
    )
    percentage_positive: float = field(
        default=0.5, metadata={'help': 'Ratio of positive (sponsor) segments to include in final output'})

    train_split: float = field(
        default=0.9, metadata={'help': 'Ratio of training data. Value between 0 and 1.'})

    # TODO play around with ratios? lower test/validation split?
    test_split: float = field(
        default=0.05, metadata={'help': 'Ratio of testing data. Value between 0 and 1.'})
    valid_split: float = field(
        default=0.05, metadata={'help': 'Ratio of validation data. Value between 0 and 1.'})

    start_index: int = field(default=None, metadata={
        'help': 'Video to start at.'})

    max_videos: int = field(default=None, metadata={
        'help': 'Maximum number of videos to preprocess.'})

    max_segments: int = field(default=None, metadata={
        'help': 'Maximum number of segments to produce to preprocess.'})

    raw_data_dir: Optional[str] = field(
        default='raw',
        metadata={
            'help': 'Raw data directory'
        },
    )
    raw_data_file: Optional[str] = field(
        default='sponsorTimes.csv',
        metadata={
            'help': 'Raw data file'
        },
    )

    min_wps: float = field(
        default=1.5, metadata={'help': 'Ignore videos with not enough words spoken per second. This is usually indicitive of video whose captions aren\'t English.'})
    # 0.1 ~ 1%
    # 0.4 ~ 2.5%
    # 0.9 ~ 5%


# Mirrors for database
MIRRORS = [
    'https://sponsor.ajay.app/database/sponsorTimes.csv',  # Latest
    'https://sb-mirror.mchang.xyz/sponsorTimes.csv',  # 5 minute delay
    'https://sb.ltn.fi/database/sponsorTimes.csv',  # 5 minute delay
]
# TODO only download latest updates/changes


def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`.

    Adapted from https://stackoverflow.com/a/42071418
    """
    chunk_size = 1024
    r = requests.get(url, stream=True)
    total_bytes = int(r.headers['Content-Length'])
    with open(filename, 'wb') as f, tqdm(unit='B', total=total_bytes) as progress:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                f.write(chunk)

    return total_bytes == os.path.getsize(filename)


def load_datasets(dataset_args):
    print('Reading datasets')
    data_files = {}

    if dataset_args.train_file is not None:
        data_files['train'] = os.path.join(
            dataset_args.data_dir, dataset_args.train_file)
    if dataset_args.validation_file is not None:
        data_files['validation'] = os.path.join(
            dataset_args.data_dir, dataset_args.validation_file)
    if dataset_args.test_file is not None:
        data_files['test'] = os.path.join(
            dataset_args.data_dir, dataset_args.test_file)

    return load_dataset('json', data_files=data_files, cache_dir=dataset_args.dataset_cache_dir)


@dataclass
class DatasetArguments:
    data_dir: Optional[str] = field(
        default='data',
        metadata={
            'help': 'The directory which stores train, test and/or validation data.'
        },
    )
    processed_file: Optional[str] = field(
        default='segments.json',
        metadata={
            'help': 'Processed data file'
        },
    )
    processed_database: Optional[str] = field(
        default='processed_database.json',
        metadata={
            'help': 'Processed database file'
        },
    )

    train_file: Optional[str] = field(
        default='train.json', metadata={'help': 'The input training data file (a jsonlines file).'}
    )
    validation_file: Optional[str] = field(
        default='valid.json',
        metadata={
            'help': 'An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines file).'
        },
    )
    test_file: Optional[str] = field(
        default='test.json',
        metadata={
            'help': 'An optional input test data file to evaluate the metrics (rouge) on (a jsonlines file).'
        },
    )
    excess_file: Optional[str] = field(
        default='excess.json',
        metadata={
            'help': 'The excess segments left after the split'
        },
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Where to store the cached datasets'
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={'help': 'Overwrite the cached training and evaluation sets'}
    )

    positive_file: Optional[str] = field(
        default='sponsor_segments.json', metadata={'help': 'File to output sponsored segments to (a jsonlines file).'}
    )
    negative_file: Optional[str] = field(
        default='normal_segments.json', metadata={'help': 'File to output normal segments to (a jsonlines file).'}
    )

    def __post_init__(self):
        # TODO check if train/validation datasets exist
        if self.train_file is None and self.validation_file is None:
            raise ValueError(
                'Need either a dataset name or a training/validation file.')


def main():
    # Responsible for getting transcrips using youtube_transcript_api,
    # then labelling it according to SponsorBlock's API

    logging.getLogger().setLevel(logging.INFO)  # TODO make param

    # Generate final.json from sponsorTimes.csv
    hf_parser = HfArgumentParser((
        PreprocessArguments,
        DatasetArguments,
        segment.SegmentationArguments,
        ModelArguments,
        GeneralArguments
    ))
    preprocess_args, dataset_args, segmentation_args, model_args, _ = hf_parser.parse_args_into_dataclasses()

    raw_dataset_path = os.path.join(
        preprocess_args.raw_data_dir, preprocess_args.raw_data_file)

    if preprocess_args.update_database:
        print('Updating database')
        for mirror in MIRRORS:
            print('Downloading from', mirror)
            if download_file(mirror, raw_dataset_path):
                break
            print('Failed, trying next')

    processed_db_path = os.path.join(
        dataset_args.data_dir, dataset_args.processed_database)

    @lru_cache(maxsize=1)
    def read_db():
        if not preprocess_args.overwrite and os.path.exists(processed_db_path):
            with open(processed_db_path) as fp:
                return json.load(fp)
        print('Processing raw database')
        db = {}

        allowed_categories = list(map(str.lower, CATGEGORY_OPTIONS))
        with open(raw_dataset_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            for line in reader:

                if line['service'] != 'YouTube':
                    continue
                if len(line['videoID']) != 11:
                    continue  # Invalid youtube video ID

                if line['category'] not in allowed_categories:
                    continue
                if line['actionType'] != 'skip':
                    continue

                # Ignore hidden items
                if line['hidden'] == '1' or line['shadowHidden'] == '1':
                    continue

                # Skip those that aren't highly voted
                votes = int(line['votes'])
                if votes < preprocess_args.min_votes:
                    continue

                locked = line['locked'] == '1'

                reputation = float(line['reputation'])
                # if reputation < preprocess_args.min_reputation:
                #     continue # TODO add back?
                # Problems like mGVn1wCkBrE

                # TODO ignore if over max_duration

                if line['videoID'] not in db:
                    db[line['videoID']] = []

                db[line['videoID']].append({
                    'uuid': line['UUID'],
                    'start': float(line['startTime']),
                    'end': float(line['endTime']),
                    'votes': votes,
                    'locked': locked,
                    'views': int(line['views']),
                    'submission_time': float(line['timeSubmitted'])/1e3,
                    'reputation': reputation,
                    'category': line['category'],
                    # 'action': line['actionType'],
                })

        # Remove duplicate sponsor segments by choosing best (most votes)
        print('Remove duplicate segments')
        for key in db:
            db[key] = remove_duplicate_segments(db[key])

        # We now remove whole videos from the list
        # Helps with obtaining "fully-labelled" videos
        min_date = datetime.strptime(preprocess_args.min_date, '%d/%m/%Y')
        max_date = datetime.strptime(preprocess_args.max_date, '%d/%m/%Y')
        for key in list(db):

            if any(datetime.fromtimestamp(x['submission_time']) < min_date for x in db[key]):
                # Remove videos where any of its segments were submitted before min_date
                # (essentially removes videos uploaded before min_date)
                # Prevents issues where some segments of a video are excluded
                del db[key]
            elif all(datetime.fromtimestamp(x['submission_time']) > max_date for x in db[key]):
                # Remove videos where all of its segments were submitted after max_date
                # (essentially removes videos uploaded after max_date)
                # Allows for segments to be corrected for past videos
                del db[key]
            elif any(not x['locked'] and x['views'] < preprocess_args.min_views for x in db[key]):
                # Remove videos where any of its non-locked segments do not have enough views
                # (essentially skips videos that have not been fully watched/reviewed)
                # Always include segments locked by VIPs, regardless of view count
                del db[key]

        print('Saved', len(db), 'videos')

        with open(processed_db_path, 'w') as fp:
            json.dump(db, fp)

        return db

    # 'videoID', 'startTime', 'endTime', 'votes', 'locked', 'incorrectVotes', 'UUID',
    # 'userID', 'timeSubmitted', 'views', 'category', 'actionType', 'service', 'videoDuration',
    # 'hidden', 'reputation', 'shadowHidden', 'hashedVideoID', 'userAgent', 'description'
    if preprocess_args.do_transcribe:
        print('Collecting videos')
        parsed_database = read_db()

        # Remove transcripts already processed
        finished = set(x.split('.')[0] for x in os.listdir(
            'transcripts/auto/') + os.listdir('transcripts/manual/'))

        video_ids = list(parsed_database.keys() - finished)

        # Create tasks generator
        tasks = (
            Task(get_words, video_id)
            for video_id in video_ids
        )

        print('Downloading transcripts')
        with tqdm(total=len(video_ids)) as progress:
            def callback(task):
                progress.set_description(f'Processing {task.args[0]}')
                progress.update()

            InterruptibleTaskPool(
                tasks, preprocess_args.num_jobs, callback).start()

    final_path = os.path.join(
        dataset_args.data_dir, dataset_args.processed_file)

    if preprocess_args.do_create:
        print('Create final data')

        final_data = {}

        parsed_database = read_db()

        # TODO parallelise?
        with tqdm(total=len(parsed_database)) as progress:
            for index, (video_id, segments) in enumerate(parsed_database.items()):
                if preprocess_args.max_videos is not None and index >= preprocess_args.max_videos:
                    break
                progress.set_description(f'Processing {video_id}')
                progress.update()

                video_words = get_words(video_id, process=False)
                if not video_words:
                    continue

                final_vid_segs = []
                for seg in segments:  # Only add segments with high enough wps
                    segment_words = segment.extract_segment(
                        video_words, seg['start'], seg['end'])

                    if len(segment_words) <= 1:
                        continue  # Useless to add segment since no words

                    # duration = segment.word_end(segment_words[-1]) - segment.word_start(segment_words[0])
                    duration = seg['end'] - seg['start']
                    wps = len(segment_words)/duration if duration > 0 else 0

                    # print(video_id, wps)
                    if wps < preprocess_args.min_wps:
                        # Skip sponsor segments without many words
                        # e.g. music ads with some words on each side
                        # progress.set_description(f'Skipping bad segment in {video_id} (wps={wps})')
                        continue
                    final_vid_segs.append(seg)

                if final_vid_segs:
                    final_data[video_id] = final_vid_segs

        # Save data
        with open(final_path, 'w') as fp:
            json.dump(final_data, fp)

        # final_data = preprocess(
        #     raw_dataset_path, final_path, preprocess_args.min_votes)
        # # TODO save metadata in final.json?

    elif os.path.exists(final_path):
        # Already exists
        logging.info(f'{final_path} exists, opening file')
        with open(final_path) as fp:
            final_data = json.load(fp)
        logging.info(f'Found {len(final_data)} videos')
    else:
        return  # Do not continue

    # TODO shuffle final_data
    # if not os.path.exists(excess_path) or preprocess_args.overwrite
    # TODO use overwrite param

    os.makedirs(dataset_args.data_dir, exist_ok=True)

    positive_file = os.path.join(
        dataset_args.data_dir, dataset_args.positive_file)
    negative_file = os.path.join(
        dataset_args.data_dir, dataset_args.negative_file)

    if preprocess_args.do_generate:
        print('Generating')
        # max_videos=preprocess_args.max_videos,
        # max_segments=preprocess_args.max_segments,
        # , max_videos, max_segments

        from model import get_model_tokenizer
        model, tokenizer = get_model_tokenizer(model_args.model_name_or_path)

        # TODO
        # count_videos = 0
        # count_segments = 0

        data = final_data.items()

        start_index = preprocess_args.start_index or 0
        end_index = (preprocess_args.max_videos or len(data)) + start_index

        data = list(itertools.islice(data, start_index, end_index))

        write_mode = 'w' if preprocess_args.overwrite else 'a'
        with open(positive_file, write_mode, encoding='utf-8') as positive, \
                open(negative_file, write_mode, encoding='utf-8') as negative, \
                tqdm(data) as progress:

            for offset, (video_id, sponsor_segments) in enumerate(data):

                progress.set_description(f'Processing {video_id}')
                progress.update()

                words = get_words(video_id, process=False)
                if not words:
                    continue

                num_words = len(words)
                if num_words <= 1:
                    continue

                # TODO only count words that aren't [Music], [Applause], etc.

                segments = segment.generate_labelled_segments(
                    words, tokenizer, segmentation_args, sponsor_segments)

                if not segments:
                    continue

                for seg in segments:
                    seg_start = segment.word_start(seg[0])
                    seg_end = segment.word_end(seg[-1])
                    # duration = seg_end - seg_start
                    # wps = len(seg)/duration if duration > 0 else 0

                    # # Ignore segments with "not enough words" in the transcript
                    # # Must do here since this includes non-sponsor segments
                    # if wps < preprocess_args.min_wps:
                    #     continue

                    d = {
                        'video_index': offset + start_index,
                        'video_id': video_id,
                        'text': ' '.join(x['cleaned'] for x in seg),
                        'start': seg_start,
                        'end': seg_end,
                    }

                    extracted_segments = extract_sponsors(seg)
                    if extracted_segments:
                        extracted_texts = []
                        for s in extracted_segments:
                            w = ' '.join(q['cleaned'] for q in s['words'])
                            category = s['category'].upper()
                            extracted_texts.append(
                                f'{START_SEGMENT_TEMPLATE.format(category)} {w} {END_SEGMENT_TEMPLATE.format(category)}'
                            )

                        d['extracted'] = f' {CustomTokens.BETWEEN_SEGMENTS.value} '.join(
                            extracted_texts)
                        print(json.dumps(d), file=positive)

                    else:
                        d['extracted'] = CustomTokens.NO_SEGMENT.value
                        print(json.dumps(d), file=negative)

    if preprocess_args.do_split:
        print('Splitting')
        print('Read files')

        with open(positive_file, encoding='utf-8') as positive:
            sponsors = positive.readlines()

        with open(negative_file, encoding='utf-8') as negative:
            non_sponsors = negative.readlines()

        print('Shuffle')
        random.shuffle(sponsors)
        random.shuffle(non_sponsors)

        print('Calculate ratios')
        # Ensure correct ratio of positive to negative segments
        percentage_negative = 1 - preprocess_args.percentage_positive

        if preprocess_args.percentage_positive * len(sponsors) > len(non_sponsors):
            # Negative is limiting
            z = int(preprocess_args.percentage_positive /
                    percentage_negative * len(non_sponsors))

            excess = sponsors[z:]
            sponsors = sponsors[:z]

        else:
            # Positive is limiting
            z = int(percentage_negative /
                    preprocess_args.percentage_positive * len(sponsors))

            excess = non_sponsors[z:]
            non_sponsors = non_sponsors[:z]

        print('Join')
        all_labelled_segments = sponsors + non_sponsors

        random.shuffle(all_labelled_segments)

        print('Split')
        ratios = [preprocess_args.train_split,
                  preprocess_args.test_split,
                  preprocess_args.valid_split]

        train_data, test_data, valid_data = split(
            all_labelled_segments, ratios)

        splits = {
            dataset_args.train_file: train_data,
            dataset_args.test_file: test_data,
            dataset_args.validation_file: valid_data
        }

        # Output training, testing and validation data
        for name, items in splits.items():
            outfile = os.path.join(dataset_args.data_dir, name)
            if not os.path.exists(outfile) or preprocess_args.overwrite:
                with open(outfile, 'w', encoding='utf-8') as fp:
                    fp.writelines(items)
            else:
                print('Skipping', name)

        print('Write')
        # Save excess items
        excess_path = os.path.join(
            dataset_args.data_dir, dataset_args.excess_file)
        if not os.path.exists(excess_path) or preprocess_args.overwrite:
            with open(excess_path, 'w', encoding='utf-8') as fp:
                fp.writelines(excess)
        else:
            print('Skipping', dataset_args.excess_file)

        print('Finished splitting:', len(sponsors),
              'sponsors,', len(non_sponsors), 'non sponsors')


def split(arr, ratios):
    """Split array according to ratios. Sum of ratios should be <= 1"""
    to_return = []

    cumulative_sum = 0
    for r in ratios:
        current = cumulative_sum
        cumulative_sum += r * len(arr)
        to_return.append(arr[int(current):int(cumulative_sum)])

    return to_return


if __name__ == '__main__':
    main()
