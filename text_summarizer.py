from transformers import pipeline


class TextSummarizer:
    """Uses the Facebook BART model"""

    def __init__(self, min_length=30, max_length=150):
        self.pipeline = pipeline(task='summarization', model='facebook/bart-large-cnn')
        self.gen_kwargs = {'max_length': max_length, 'min_length': min_length, 'do_sample': False}

    def summarize(self, input_text):
        """Returns a summarized text using the model initialized"""
        return self.pipeline(input_text, **self.gen_kwargs)[0]['summary_text']
