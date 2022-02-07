from utils import jaccard
from functools import lru_cache
from datetime import datetime
import itertools
from typing import Optional, List
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


PROFANITY_RAW = '[ __ ]'  # How YouTube transcribes profanity
PROFANITY_CONVERTED = '*****'  # Safer version for tokenizing


NUM_DECIMALS = 3


def parse_transcript_json(json_data, granularity):
    assert json_data['wireMagic'] == 'pb3'

    assert granularity in ('word', 'chunk')

    # TODO remove bracketed words?
    # (kiss smacks)
    # (upbeat music)
    # [text goes here]

    # Some manual transcripts aren't that well formatted... but do have punctuation
    # https://www.youtube.com/watch?v=LR9FtWVjk2c

    parsed_transcript = []

    events = json_data['events']

    for event_index, event in enumerate(events):
        segments = event.get('segs')
        if not segments:
            continue

        # This value is known (when phrase appears on screen)
        start_ms = event['tStartMs']
        total_characters = 0

        new_segments = []
        for seg in segments:
            text = seg['utf8'].replace('\n', ' ').replace(
                PROFANITY_RAW, PROFANITY_CONVERTED,  # Needed for auto-generated transcripts
            ).strip()
            if not text:
                continue

            offset_ms = seg.get('tOffsetMs', 0)

            new_segments.append({
                'text': text,
                'start': round((start_ms + offset_ms)/1000, NUM_DECIMALS)
            })

            total_characters += len(text)

        if not new_segments:
            continue

        if event_index < len(events) - 1:
            next_start_ms = events[event_index + 1]['tStartMs']
            total_event_duration_ms = min(
                event.get('dDurationMs', float('inf')), next_start_ms - start_ms)
        else:
            total_event_duration_ms = event.get('dDurationMs', 0)

        avg_seconds_per_character = (
            total_event_duration_ms/total_characters)/1000

        num_char_count = 0
        for seg_index, seg in enumerate(new_segments):
            num_char_count += len(seg['text'])

            # Estimate segment end
            seg_end = seg['start'] + \
                (num_char_count * avg_seconds_per_character)

            if seg_index < len(new_segments) - 1:
                # Do not allow longer than next
                seg_end = min(seg_end, new_segments[seg_index+1]['start'])

            seg['end'] = round(seg_end, NUM_DECIMALS)
            parsed_transcript.append(seg)

    final_parsed_transcript = []
    for i in range(len(parsed_transcript)):

        word_level = granularity == 'word'
        if word_level:
            split_text = parsed_transcript[i]['text'].split()
        elif granularity == 'chunk':
            # Split on space after punctuation
            split_text = re.split(
                r'(?<=[.!?,-;])\s+', parsed_transcript[i]['text'])
            if len(split_text) == 1:
                split_on_whitespace = parsed_transcript[i]['text'].split()

                if len(split_on_whitespace) >= 8:  # Too many words
                    # Rather split on whitespace instead of punctuation
                    split_text = split_on_whitespace
                else:
                    word_level = True
        else:
            raise ValueError('Unknown granularity')

        segment_end = parsed_transcript[i]['end']
        if i < len(parsed_transcript) - 1:
            segment_end = min(segment_end, parsed_transcript[i+1]['start'])

        segment_duration = segment_end - parsed_transcript[i]['start']

        num_chars_in_text = sum(map(len, split_text))

        num_char_count = 0
        current_offset = 0
        for s in split_text:
            num_char_count += len(s)

            next_offset = (num_char_count/num_chars_in_text) * segment_duration

            word_start = round(
                parsed_transcript[i]['start'] + current_offset, NUM_DECIMALS)
            word_end = round(
                parsed_transcript[i]['start'] + next_offset, NUM_DECIMALS)

            # Make the reasonable assumption that min wps is 1.5
            final_parsed_transcript.append({
                'text': s,
                'start': word_start,
                'end': min(word_end, word_start + 1.5) if word_level else word_end
            })
            current_offset = next_offset

    return final_parsed_transcript


def list_transcripts(video_id):
    try:
        return YouTubeTranscriptApi.list_transcripts(video_id)
    except json.decoder.JSONDecodeError:
        return None


WORDS_TO_REMOVE = [
    CustomTokens.MUSIC.value,
    CustomTokens.APPLAUSE.value,
    CustomTokens.LAUGHTER.value
]


@lru_cache(maxsize=16)
def get_words(video_id, process=True, transcript_type='auto', fallback='manual', filter_words_to_remove=True, download=False, granularity='word'):
    """Get parsed video transcript with caching system
    returns None if not processed yet and process is False
    """
    # NOTE: granularity='chunk' should only be used for generating training data... nowhere else

    transcript_path = os.path.join(  # TODO use relative path to this
        'transcripts', transcript_type, f'{video_id}.json')

    raw_transcript_json = None
    try:
        if not download and os.path.exists(transcript_path):  # Load from file
            with open(transcript_path) as fp:
                raw_transcript_json = json.load(fp)  # May be empty

        elif process:
            transcript_list = list_transcripts(video_id)

            if transcript_list is not None:
                if transcript_type == 'manual':
                    ts = transcript_list.find_manually_created_transcript(
                        ['en-GB', 'en-US', 'en'])
                else:
                    ts = transcript_list.find_generated_transcript(['en'])

                raw_transcript_json = ts._http_client.get(
                    f'{ts._url}&fmt=json3').json()

    except (TooManyRequests, YouTubeRequestFailed):
        raise  # Cannot recover from these errors and do not mark as empty transcript

    except requests.exceptions.RequestException:  # Can recover
        time.sleep(10)  # Timeout
        return get_words(video_id, process, transcript_type, fallback, granularity)

    except CouldNotRetrieveTranscript:  # Retrying won't solve
        pass  # Mark as empty transcript

    except json.decoder.JSONDecodeError:
        print('JSONDecodeError for', video_id)
        if os.path.exists(transcript_path):
            os.remove(transcript_path)  # Remove file and try again
        return get_words(video_id, process, transcript_type, fallback, granularity)

    # Tried to process it, but it was empty...
    if download or (process and not os.path.exists(transcript_path)):
        with open(transcript_path, 'w') as fp:
            json.dump(raw_transcript_json, fp)

    if not raw_transcript_json and fallback is not None:
        return get_words(video_id, process, transcript_type=fallback, fallback=None, granularity=granularity)

    if raw_transcript_json:
        processed_transcript = parse_transcript_json(
            raw_transcript_json, granularity)
        if filter_words_to_remove:
            processed_transcript = list(
                filter(lambda x: x['text'] not in WORDS_TO_REMOVE, processed_transcript))
    else:
        processed_transcript = raw_transcript_json  # Either None or []

    return processed_transcript


# TODO make min_sponsor_segment_length param
# TODO rename to extract_segments
def extract_sponsors(words, min_sponsor_segment_length=3):
    if not words:
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
        default='02/02/2022',
        metadata={'help': 'Only use videos that have some segment from before this date (exclusive). This allows for videos to have segments be corrected, but ignores new videos (posted after this date) to enter the pool.'})

    keep_duplicate_segments: bool = field(
        default=False, metadata={'help': 'Keep duplicate segments'}
    )

    do_process_database: bool = field(
        default=False, metadata={'help': 'Process the raw database'}
    )
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

    os.makedirs(dataset_args.data_dir, exist_ok=True)
    processed_db_path = os.path.join(
        dataset_args.data_dir, dataset_args.processed_database)

    # TODO process all valid possible items and then do filtering only later
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

                # Never show:
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
        if not preprocess_args.keep_duplicate_segments:
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

            # TODO remove videos that contain a full-video label?

        print('Saved', len(db), 'videos')

        with open(processed_db_path, 'w') as fp:
            json.dump(db, fp)

        return db

    if preprocess_args.do_process_database:
        read_db()

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

        # https://stackoverflow.com/a/63495323
        import concurrent
        POLL_INTERVAL = 0.1

        # Wrap get words function to return video_id after completion
        def get_words_wrapper(video_id):
            get_words(video_id)
            return video_id

        print('Setting up ThreadPoolExecutor')
        with concurrent.futures.ThreadPoolExecutor(max_workers=preprocess_args.num_jobs) as pool, \
                tqdm(total=len(video_ids)) as progress:

            all_futures = (pool.submit(get_words_wrapper, video_id)
                           for video_id in video_ids)
            to_process = set(itertools.islice(
                all_futures, preprocess_args.num_jobs))
            try:
                while to_process:
                    just_finished, to_process = concurrent.futures.wait(
                        to_process, timeout=POLL_INTERVAL)
                    to_process |= set(itertools.islice(
                        all_futures, len(just_finished)))

                    for d in just_finished:
                        progress.set_description(f'Processed {d.result()}')
                        progress.update()

            except KeyboardInterrupt:
                print('Gracefully shutting down: Cancelling unscheduled tasks')

                # only futures that are not done will prevent exiting
                for future in to_process:
                    future.cancel()

                print('Waiting for in-progress tasks to complete')
                concurrent.futures.wait(to_process, timeout=None)
                print('Cancellation successful')

    final_path = os.path.join(
        dataset_args.data_dir, dataset_args.processed_file)

    if preprocess_args.do_create:
        print('Create final data')

        final_data = {}

        parsed_database = read_db()

        transcribed = set(x.split('.')[0] for x in os.listdir(
            'transcripts/auto/') + os.listdir('transcripts/manual/'))

        # Only consider videos that have been transcribed already
        video_ids = parsed_database.keys() & transcribed

        with tqdm(total=len(video_ids)) as progress:
            for index, video_id in enumerate(video_ids):
                if preprocess_args.max_videos is not None and index >= preprocess_args.max_videos:
                    break
                progress.set_description(f'Processing {video_id}')
                progress.update()

                video_words = get_words(video_id, process=False)
                if not video_words:
                    continue

                final_vid_segs = []
                # Only add segments with high enough wps
                for seg in parsed_database[video_id]:
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

        write_mode = 'w'  # if preprocess_args.overwrite else 'a'
        with open(positive_file, write_mode, encoding='utf-8') as positive, \
                open(negative_file, write_mode, encoding='utf-8') as negative, \
                tqdm(data) as progress:

            for offset, (video_id, sponsor_segments) in enumerate(data):

                progress.set_description(f'Processing {video_id}')
                progress.update()

                # Use chunk granularity to improve manual transcripts
                words = get_words(video_id, process=False, granularity='chunk')
                if not words:
                    continue

                if len(words) <= 1:
                    continue

                segments = segment.generate_labelled_segments(
                    words, tokenizer, segmentation_args, sponsor_segments)

                if not segments:
                    continue

                for seg in segments:
                    seg_start = segment.word_start(seg[0])
                    seg_end = segment.word_end(seg[-1])
                    duration = seg_end - seg_start
                    wps = len(seg)/duration if duration > 0 else 0

                    # Ignore segments with "not enough words" in the transcript
                    # Must do here since this includes non-sponsor segments
                    if wps < preprocess_args.min_wps:
                        continue

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
