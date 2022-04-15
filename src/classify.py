from transformers import TextClassificationPipeline
import preprocess
import segment


class SponsorBlockClassificationPipeline(TextClassificationPipeline):
    def __init__(self, model, tokenizer):
        device = next(model.parameters()).device.index
        super().__init__(model=model, tokenizer=tokenizer,
                         return_all_scores=True, truncation=True, device=device)

    def preprocess(self, data, **tokenizer_kwargs):
        # TODO add support for lists
        texts = []

        if not isinstance(data, list):
            data = [data]

        for d in data:
            if isinstance(d, dict):  # Otherwise, get data from transcript
                words = preprocess.get_words(d['video_id'])
                segment_words = segment.extract_segment(
                    words, d['start'], d['end'])
                text = preprocess.clean_text(
                    ' '.join(x['text'] for x in segment_words))
                texts.append(text)
            elif isinstance(d, str):  # If string, assume this is what user wants to classify
                texts.append(d)
            else:
                raise ValueError(f'Invalid input type: "{type(d)}"')

        return self.tokenizer(
            texts, return_tensors=self.framework, **tokenizer_kwargs)


def main():
    pass


if __name__ == '__main__':
    main()
