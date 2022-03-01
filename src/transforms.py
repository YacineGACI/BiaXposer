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
            "Who " + input + "?",
            "There are " + self.group_token + "."
        )



class MaskedLanguageModelingTemplateTransformer(TemplateTransformer):
    def transform(self, input):
        return (self.group_token + " " + input,)




class MultipleChoiceTemplateTransformer(TemplateTransformer):
    def transform(self, input):
        return (
            "They " + input + ".",
            "That's because they are " + self.group_token + "."
        )