
from math import ceil, floor
import streamlit.components.v1 as components
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import streamlit as st
import sys
import os
import json
from urllib.parse import quote
from huggingface_hub import hf_hub_download

# Allow direct execution
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))  # noqa

from predict import SegmentationArguments, ClassifierArguments, predict as pred, seconds_to_time  # noqa
from evaluate import EvaluationArguments
from shared import device

st.set_page_config(
    page_title="SponsorBlock ML",
    page_icon="🤖",
    #  layout='wide',
    #  initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/xenova/sponsorblock-ml',
        'Report a bug': 'https://github.com/xenova/sponsorblock-ml/issues/new/choose',
        #  'About': "# This is a header. This is an *extremely* cool app!"
    }
)

MODEL_PATH = 'Xenova/sponsorblock-small_v2022.01.19'

CLASSIFIER_PATH = 'Xenova/sponsorblock-classifier'


@st.cache(allow_output_mutation=True)
def persistdata():
    return {}


# Faster caching system for predictions (No need to hash)
predictions_cache = persistdata()


@st.cache(allow_output_mutation=True)
def load_predict():
    # Use default segmentation and classification arguments
    evaluation_args = EvaluationArguments(model_path=MODEL_PATH)
    segmentation_args = SegmentationArguments()
    classifier_args = ClassifierArguments()

    model = AutoModelForSeq2SeqLM.from_pretrained(evaluation_args.model_path)
    model.to(device())

    tokenizer = AutoTokenizer.from_pretrained(evaluation_args.model_path)

    # Save classifier and vectorizer
    hf_hub_download(repo_id=CLASSIFIER_PATH,
                    filename=classifier_args.classifier_file,
                    cache_dir=classifier_args.classifier_dir,
                    force_filename=classifier_args.classifier_file,
                    )
    hf_hub_download(repo_id=CLASSIFIER_PATH,
                    filename=classifier_args.vectorizer_file,
                    cache_dir=classifier_args.classifier_dir,
                    force_filename=classifier_args.vectorizer_file,
                    )

    def predict_function(video_id):
        if video_id not in predictions_cache:
            predictions_cache[video_id] = pred(
                video_id, model, tokenizer,
                segmentation_args=segmentation_args,
                classifier_args=classifier_args
            )
        return predictions_cache[video_id]

    return predict_function


CATGEGORY_OPTIONS = {
    'SPONSOR': 'Sponsor',
    'SELFPROMO': 'Self/unpaid promo',
    'INTERACTION': 'Interaction reminder',
}


# Load prediction function
predict = load_predict()


def main():

    # Display heading and subheading
    st.write('# SponsorBlock ML')
    st.write('##### Automatically detect in-video YouTube sponsorships, self/unpaid promotions, and interaction reminders.')

    # Load widgets
    video_id = st.text_input('Video ID:')  # , placeholder='e.g., axtQvkSpoto'

    categories = st.multiselect('Categories:',
                                CATGEGORY_OPTIONS.keys(),
                                CATGEGORY_OPTIONS.keys(),
                                format_func=CATGEGORY_OPTIONS.get
                                )

    # Hide segments with a confidence lower than
    confidence_threshold = st.slider(
        'Confidence Threshold (%):', min_value=0, max_value=100)

    video_id_length = len(video_id)
    if video_id_length == 0:
        return

    elif video_id_length != 11:
        st.exception(ValueError('Invalid YouTube ID'))
        return

    with st.spinner('Running model...'):
        predictions = predict(video_id)

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

    json_data = quote(json.dumps(submit_segments))
    link = f'[Submit Segments](https://www.youtube.com/watch?v={video_id}#segments={json_data})'
    st.markdown(link, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
