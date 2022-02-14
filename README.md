---
title: Sponsorblock ML
emoji: 🤖
colorFrom: yellow
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: true
---

# SponsorBlock-ML
Automatically detect in-video YouTube sponsorships, self/unpaid promotions, and interaction reminders. The model was trained using the [SponsorBlock](https://sponsor.ajay.app/) [database](https://sponsor.ajay.app/database) licensed used under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Check out the online demo application at [https://xenova.github.io/sponsorblock-ml/](https://xenova.github.io/sponsorblock-ml/), or follow the instructions below to run it locally.

---
## Installation

1. Download the repository:
	```bash
	git clone https://github.com/xenova/sponsorblock-ml.git
	cd sponsorblock-ml
	```

2. Install the necessary dependencies:
	```bash
	pip install -r requirements.txt
	```

3. Run the application:
	```bash
	streamlit run app.py
	```
## Predicting

- Predict for a single video using the `--video_id` argument. For example:
	```bash
	python src/predict.py --video_id zo_uoFI1WXM
	```

- Predict for multiple videos using the `--video_ids` argument. For example:
	```bash
	python src/predict.py --video_ids IgF3OX8nT0w ao2Jfm35XeE
	```

- Predict for a whole channel using the `--channel_id` argument. For example:

	```bash
	python src/predict.py --channel_id UCHnyfMqiRRG1u-2MsSQLbXA
	```

Note that on the first run, the program will download the necessary models (which may take some time).

		
---

## Evaluating

### Measuring Accuracy
This is primarly used to measure the accuracy (and other metrics) of the model (defaults to [Xenova/sponsorblock-small](https://huggingface.co/Xenova/sponsorblock-small)).
```bash
python src/evaluate.py
```
In addition to the calculated metrics, missing and incorrect segments are output, allowing for improvements to be made to the database:
- Missing segments: Segments which were predicted by the model, but are not in the database.
- Incorrect segments: Segments which are in the database, but the model did not predict (meaning that the model thinks those segments are incorrect).

### Moderation
This can also be used to moderate parts of the database. To moderate the whole database, first run:
```bash
python src/preprocess.py --do_process_database --processed_database whole_database.json --min_votes -1 --min_views 0 --min_date 01/01/2000 --max_date 01/01/9999 --keep_duplicate_segments
```

followed by
```bash
python src/evaluate.py --processed_file data/whole_database.json
```

The `--video_ids` and `--channel_id` arguments can also be used here. Remember to keep your database and processed database file up-to-date before running evaluations.

---

## Training
### Preprocessing

1. Download the SponsorBlock database
	```bash
	python src/preprocess.py --update_database
	```

2. Preprocess the database and generate training, testing and validation data

	```bash
	python src/preprocess.py --do_transcribe --do_create --do_generate --do_split --model_name_or_path Xenova/sponsorblock-small
	```


	1. `--do_transcribe` - Downloads and parses the transcripts from YouTube.
	2. `--do_create` - Process the database (removing unwanted and duplicate segments) and create the labelled dataset.
	3. `--do_generate` - Using the downloaded transcripts and labelled segment data, extract positive (sponsors, unpaid/self-promos and interaction reminders) and negative (normal video content) text segments and create large lists of input and target texts.
	4. `--do_split` - Using the generated positive and negative segments, split them into training, validation and testing sets (according to the specified ratios).

	Each of the above steps can be run independently (as separate commands, e.g. `python src/preprocess.py --do_transcribe`), but should be performed in order.

	For more advanced preprocessing options, run `python src/preprocess.py --help`

### Transformer 
The transformer is used to extract relevent segments from the transcript and apply a preliminary classification to the extracted text. To start finetuning from the current checkpoint, run:

```bash
python src/train.py --model_name_or_path Xenova/sponsorblock-small
```

If you wish to finetune an original transformer model, use one of the supported models (*t5-small*, *t5-base*, *t5-large*, *t5-3b*, *t5-11b*,  *google/t5-v1_1-small*, *google/t5-v1_1-base*, *google/t5-v1_1-large*, *google/t5-v1_1-xl*, *google/t5-v1_1-xxl*) as the `--model_name_or_path`. For more information, check out the relevant documentation ([t5](https://huggingface.co/docs/transformers/model_doc/t5) or [t5v1.1](https://huggingface.co/docs/transformers/model_doc/t5v1.1)).
	


### Classifier
The classifier is used to add probabilities to the category predictions. Train the classifier using:
```bash
python src/train.py --do_train_classifier --skip_train_transformer
```
