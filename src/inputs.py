class InputProcessor:
    """
        Tokenizes the Input according to the task model.
        Returns a dictionary of input_ids, attention_masks and token_type_ids
    """
    def __init__(self, tokenizer, input_names, truncation=True, padding="do_not_pad"):
        self.tokenizer = tokenizer
        self.input_names = input_names
        self.truncation = truncation
        self.padding = padding


    def tokenize(self, input):
        raise NotImplementedError







class SingleInputProcessor(InputProcessor):
    """
        Tokenizes inputs consisting of a single sentence. suitable for tasks such as Sentiment classification, hate speech detection, correct grammar detection...
    """
    def tokenize(self, input, input_names=None):
        input_names = input_names if input_names is not None else self.input_names
        return self.tokenizer(input[input_names[0]], truncation=self.truncation)








class DoubleInputProcessor(InputProcessor):
    """
        Tokenizes inputs consisting of two sentences, e.g. Textual Entailment, Paraphrase Detection, Question Answering...
    """
    def tokenize(self, input):
        input_names = input_names if input_names is not None else self.input_names
        return self.tokenizer(input[input_names[0]], input[input_names[1]], truncation=self.truncation)







##### To re-implement
class MultipleChoiceInputProcessor(InputProcessor):
    """
        Toeknizes inputs consisting of a context, and multiple choices.
    """
    def __init__(self, tokenizer, truncation=True):
        super().__init__(tokenizer, truncation, padding='longest')

    def tokenize(self, input):
        input_1 = input[0]
        choices = input[1]
        input_1 = [input_1] * len(choices)
        return self.tokenizer(input_1, choices, truncation=self.truncation, padding=self.padding)
