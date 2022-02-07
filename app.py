
from functools import partial
from math import ceil, floor
import streamlit.components.v1 as components
import streamlit as st
import sys
import os
import json
from urllib.parse import quote

# Allow direct execution
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))  # noqa

from preprocess import get_words
from predict import SegmentationArguments, ClassifierArguments, predict as pred
from evaluate import EvaluationArguments
from shared import seconds_to_time, CATGEGORY_OPTIONS
from utils import regex_search
from model import get_model_tokenizer
from errors import TranscriptError

st.set_page_config(
    page_title='SponsorBlock ML',
    page_icon='🤖',
    #  layout='wide',
    #  initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/xenova/sponsorblock-ml',
        'Report a bug': 'https://github.com/xenova/sponsorblock-ml/issues/new/choose',
        #  'About': "# This is a header. This is an *extremely* cool app!"
    }
)


YT_VIDEO_REGEX = r'''(?x)^
                (?:
                    # http(s):// or protocol-independent URL
                    (?:https?://|//)
                    (?:(?:(?:(?:\w+\.)?[yY][oO][uU][tT][uU][bB][eE](?:-nocookie|kids)?\.com/|
                    youtube\.googleapis\.com/)                        # the various hostnames, with wildcard subdomains
                    (?:.*?\#/)?                                          # handle anchor (#/) redirect urls
                    (?:                                                  # the various things that can precede the ID:
                        # v/ or embed/ or e/
                        (?:(?:v|embed|e)/(?!videoseries))
                        |(?:                                             # or the v= param in all its forms
                            # preceding watch(_popup|.php) or nothing (like /?v=xxxx)
                            (?:(?:watch|movie)(?:_popup)?(?:\.php)?/?)?
                            (?:\?|\#!?)                                  # the params delimiter ? or # or #!
                            # any other preceding param (like /?s=tuff&v=xxxx or ?s=tuff&amp;v=V36LpHqtcDY)
                            (?:.*?[&;])??
                            v=
                        )
                    ))
                    |(?:
                    youtu\.be                                        # just youtu.be/xxxx
                    )/)
                )?                                                       # all until now is optional -> you can pass the naked ID
                # here is it! the YouTube video ID
                (?P<id>[0-9A-Za-z_-]{11})'''

# https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints
# https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#experimental-t5-pre-trained-model-checkpoints

# https://huggingface.co/docs/transformers/model_doc/t5
# https://huggingface.co/docs/transformers/model_doc/t5v1.1


# Faster caching system for predictions (No need to hash)
@st.cache(persist=True, allow_output_mutation=True)
def create_prediction_cache():
    return {}


@st.cache(persist=True, allow_output_mutation=True)
def create_function_cache():
    return {}


prediction_cache = create_prediction_cache()
prediction_function_cache = create_function_cache()

MODELS = {
    'Small (293 MB)': {
        'pretrained': 'google/t5-v1_1-small',
        'repo_id': 'Xenova/sponsorblock-small',
        'num_parameters': '77M'
    },
    'Base v1 (850 MB)': {
        'pretrained': 't5-base',
        'repo_id': 'Xenova/sponsorblock-base-v1',
        'num_parameters': '220M'
    },

    'Base v1.1 (944 MB)': {
        'pretrained': 'google/t5-v1_1-base',
        'repo_id': 'Xenova/sponsorblock-base-v1.1',
        'num_parameters': '250M'
    }
}

# Create per-model cache
for m in MODELS:
    if m not in prediction_cache:
        prediction_cache[m] = {}


CLASSIFIER_PATH = 'Xenova/sponsorblock-classifier'


TRANSCRIPT_TYPES = {
    'AUTO_MANUAL': {
        'label': 'Auto-generated (fallback to manual)',
        'type': 'auto',
        'fallback': 'manual'
    },
    'MANUAL_AUTO': {
        'label': 'Manual (fallback to auto-generated)',
        'type': 'manual',
        'fallback': 'auto'
    },
    # 'TRANSLATED': 'Translated to English' # Coming soon
}


def predict_function(model_id, model, tokenizer, segmentation_args, classifier_args, video_id, words, ts_type_id):
    cache_id = f'{video_id}_{ts_type_id}'

    if cache_id not in prediction_cache[model_id]:
        prediction_cache[model_id][cache_id] = pred(
            video_id, model, tokenizer,
            segmentation_args=segmentation_args,
            classifier_args=classifier_args,
            words=words
        )
    return prediction_cache[model_id][cache_id]


def load_predict(model_id):
    model_info = MODELS[model_id]

    if model_id not in prediction_function_cache:
        # Use default segmentation and classification arguments
        evaluation_args = EvaluationArguments(model_path=model_info['repo_id'])
        segmentation_args = SegmentationArguments()
        classifier_args = ClassifierArguments(
            min_probability=0)  # Filtering done later

        model, tokenizer = get_model_tokenizer(evaluation_args.model_path)

        prediction_function_cache[model_id] = partial(
            predict_function, model_id, model, tokenizer, segmentation_args, classifier_args)

    return prediction_function_cache[model_id]


def create_button(text, url):
    return f"""<div class="row-widget stButton" style="text-align: center">
        <a href="{url}" target="_blank" rel="noopener noreferrer" class="btn-link">
            <button kind="primary" class="btn">{text}</button>
        </a>
    </div>"""


def main():
    st.markdown("""<style>
    .btn {
        display: inline-flex;
        -webkit-box-align: center;
        align-items: center;
        -webkit-box-pack: center;
        justify-content: center;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        margin: 0px;
        line-height: 1.5;
        color: inherit;
        width: auto;
        user-select: none;
        background-color: rgb(255, 255, 255);
        border: 1px solid rgba(49, 51, 63, 0.2);
    }
    .btn-link {
        color: inherit;
        text-decoration: none;
    }
    </style>""", unsafe_allow_html=True)

    top = st.container()
    output = st.empty()

    # Display heading and subheading
    top.markdown('# SponsorBlock ML')
    top.markdown(
        '##### Automatically detect in-video YouTube sponsorships, self/unpaid promotions, and interaction reminders.')

    # Add controls

    col1, col2 = top.columns(2)

    with col1:
        model_id = st.selectbox(
            'Select model', MODELS.keys(), index=0, on_change=output.empty)

    with col2:
        ts_type_id = st.selectbox(
            'Transcript type', TRANSCRIPT_TYPES.keys(), index=0, format_func=lambda x: TRANSCRIPT_TYPES[x]['label'], on_change=output.empty)

    video_input = top.text_input('Video URL/ID:', on_change=output.empty)
    categories = top.multiselect('Categories:',
                                 CATGEGORY_OPTIONS.keys(),
                                 CATGEGORY_OPTIONS.keys(),
                                 format_func=CATGEGORY_OPTIONS.get, on_change=output.empty
                                 )

    # Hide segments with a confidence lower than
    confidence_threshold = top.slider(
        'Confidence Threshold (%):', min_value=0, value=50, max_value=100, on_change=output.empty)

    if len(video_input) == 0:  # No input, do not continue
        return

    # Load prediction function
    with st.spinner('Loading model...'):
        predict = load_predict(model_id)

    with output.container():  # Place all content in output container
        video_id = regex_search(video_input, YT_VIDEO_REGEX)
        if video_id is None:
            st.exception(ValueError('Invalid YouTube URL/ID'))
            return

        try:
            with st.spinner('Downloading transcript...'):
                words = get_words(video_id,
                                  transcript_type=TRANSCRIPT_TYPES[ts_type_id]['type'],
                                  fallback=TRANSCRIPT_TYPES[ts_type_id]['fallback']
                                  )
        except TranscriptError:
            pass

        if not words:
            st.error('No transcript found!')
            return

        with st.spinner('Running model...'):
            predictions = predict(video_id, words, ts_type_id)

        if len(predictions) == 0:
            st.success('No segments found!')
            return

        submit_segments = []
        for index, prediction in enumerate(predictions, start=1):
            if prediction['category'] not in categories:
                continue  # Skip

            confidence = prediction['probability'] * 100

            if confidence < confidence_threshold:
                continue

            submit_segments.append({
                'segment': [prediction['start'], prediction['end']],
                'category': prediction['category'].lower(),
                'actionType': 'skip'
            })
            start_time = seconds_to_time(prediction['start'])
            end_time = seconds_to_time(prediction['end'])
            with st.expander(
                f"[{prediction['category']}] Prediction #{index} ({start_time} \u2192 {end_time})"
            ):

                url = f"https://www.youtube-nocookie.com/embed/{video_id}?&start={floor(prediction['start'])}&end={ceil(prediction['end'])}"
                # autoplay=1controls=0&&modestbranding=1&fs=0

                # , width=None, height=None, scrolling=False
                components.iframe(url, width=670, height=376)

                text = ' '.join(w['text'] for w in prediction['words'])
                st.write(f"**Times:** {start_time} \u2192 {end_time}")
                st.write(
                    f"**Category:** {CATGEGORY_OPTIONS[prediction['category']]}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.write(f'**Text:** "{text}"')

        if len(submit_segments) == 0:
            st.success(
                f'No segments found! ({len(predictions)} ignored due to filters/settings)')
            return

        num_hidden = len(predictions) - len(submit_segments)
        if num_hidden > 0:
            st.info(
                f'{num_hidden} predictions hidden (adjust the settings and filters to view them all).')

        json_data = quote(json.dumps(submit_segments))
        link = f'https://www.youtube.com/watch?v={video_id}#segments={json_data}'
        st.markdown(create_button('Submit Segments', link),
                    unsafe_allow_html=True)

        st.markdown(f"""<div style="text-align: center;font-size: 16px;margin-top: 6px">
        <a href="https://wiki.sponsor.ajay.app/w/Automating_Submissions" target="_blank" rel="noopener noreferrer">(Review before submitting!)</a>
        </div>""", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
