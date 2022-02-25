class TemplateTransformer:
    def __init__(self, group_token="<Group>"):
        self.group_token = group_token

    def transform(self, input):
        raise NotImplementedError




class SingleInputSequenceClassificationTemplateTransformer(TemplateTransformer):
    def transform(self, input):
        return (self.group_token + " " + input,)





class DoubleInputSequenceClassificationTemplateTransformer(TemplateTransformer):
    def transform(self, input):
        return (
            "They " + input,
            self.group_token + " " + input
        )





class QuestionAnsweringTemplateTransformer(TemplateTransformer):
    def transform(self, input):
        return (
            "There are " + self.group_token + ".",
            "Who " + input + "?"
        )



class MaskedLanguageModelingTemplateTransformer(TemplateTransformer):
    def transform(self, input):
        return (self.group_token + " " + input,)