from transformers import AutoTokenizer





class InputProcessor:
    """
        Tokenizes the Input according to the task model.
        Returns a dictionary of input_ids, attention_masks and token_type_ids
    """
    def __init__(self, tokenizer_name, truncation=True):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.truncarion = truncation


    def tokenize(self, input):
        raise NotImplementedError







class SingleInputProcessor(InputProcessor):
    """
        Tokenizes inputs consisting of a single sentence. suitable for tasks such as Sentiment classification, hate speech detection, correct grammar detection...
    """
    def __init__(self, tokenizer_name, truncation=True):
        super().__init__(tokenizer_name, truncation)

    def tokenize(self, input):
        return self.tokenizer(input, truncation=self.truncation)








class DoubleInputProcessor(InputProcessor):
    """
        Tokenizes inputs consisting of two sentences, e.g. Textual Entailment, Paraphrase Detection, Question Answering...
    """
    def __init__(self, tokenizer_name, truncation=True):
        super().__init__(tokenizer_name, truncation)

    def tokenize(self, input_1, input_2):
        return self.tokenizer(input_1, input_2, truncation=self.truncation)








class MultipleChoiceInputProcessor(InputProcessor):
    """
        Toeknizes inputs consisting of a context, and multiple choices.
    """
    def __init__(self, tokenizer_name, truncation=True):
        super().__init__(tokenizer_name, truncation)

    def tokenize(self, input_1, choices):
        input_1 = [input_1] * len(choices)
        return self.tokenizer(input_1, choices, truncation=self.truncation)
