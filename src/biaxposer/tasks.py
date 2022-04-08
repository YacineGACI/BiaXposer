import tqdm

import torch

from src.inputs import SingleInputProcessor, DoubleInputProcessor, MultipleChoiceInputProcessor
from src.outputs import SoftmaxOutputProcessor, MaskedLanguageModelingOutputProcessor, QuestionAsnweringOutputProcessor, MultipleChoiceOutputProcessor
from src.transforms import SingleInputSequenceClassificationTemplateTransformer, DoubleInputSequenceClassificationTemplateTransformer, MaskedLanguageModelingTemplateTransformer, QuestionAnsweringTemplateTransformer, MultipleChoiceTemplateTransformer
from src.utils import zip_longest_with_cycling



class TaskOutput:
    def __init__(self, output, sentence_id, def_word, group, bias_type, gold_label):
        self.output = output
        self.sentence_id = sentence_id
        self.def_word = def_word
        self.group = group
        self.bias_type = bias_type
        self.gold_label = gold_label


    def __str__(self):
        return '''
            output: {},
            id: {},
            def_word: {},
            group: {},
            bias_type: {},
            gold_label: {}
        
        '''.format(self.output, self.sentence_id, self.def_word, self.group, self.bias_type, self.gold_label)


    def __repr__(self):
        return self.__str__()





class Task:
    def __init__(self, model, bias_types, templates, group_token, label_name, input_processor=None, output_processor=None, no_cuda=False):
        self.model = model
        self.bias_types = bias_types
        self.templates = templates
        self.input_processor = input_processor
        self.output_processor = output_processor
        self.group_token = group_token
        self.label_name = label_name
        self.device = 'cuda' if not no_cuda and torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)


    
    def replace_mask(self, template, mask, replacement):
        return template.replace(mask, replacement)




    def run(self):
        raise NotImplementedError

















class SequenceClassificationTask(Task):
    
    def run(self):

        scores = []
        
        for template_id, template in tqdm.tqdm(enumerate(self.templates)):

            for bias_type in self.bias_types:

                for group in bias_type.groups.values():

                    for def_word in group.definition_words:
                        
                        # Replace the <Group> token with current definition word
                        input = {k:v if k not in self.input_processor.input_names else self.replace_mask(v, self.group_token, def_word) for k, v in template.items()}

                        # Tokenize the input
                        input = self.input_processor.tokenize(input)

                        # Make tensors and batches of 1
                        input = {k: torch.tensor(v).unsqueeze(0).to(self.device) for k, v in input.items()}

                        # Use the model for predictions
                        output = self.model(**input)

                        # Process the output
                        logits = self.output_processor.process_output(output)

                        scores.append(
                            TaskOutput(
                                output=logits.squeeze().tolist(),
                                sentence_id=template_id,
                                def_word=def_word,
                                group=group.group_name,
                                bias_type=bias_type.bias_type_name,
                                gold_label=template[self.label_name]
                            )
                        )
        
        return scores
            









class LanguageModelingTask(Task):

    def __init__(self, model, bias_types, templates, group_token, label_name, input_processor=None, output_processor=None, no_cuda=False, topk=5):
        super().__init__(model, bias_types, templates, group_token, label_name, input_processor, output_processor, no_cuda)
        self.topk = topk


    def get_mask_position(self, input_ids):
        "Finds the position of the [MASK] in the input"
        mask_id = self.input_processor.tokenizer.convert_tokens_to_ids(self.input_processor.tokenizer.mask_token)
        batch_index, token_index = (input_ids == mask_id).nonzero(as_tuple=True)
        return batch_index.item(), token_index.item()



    def run(self):

        scores = []
        
        for template_id, template in tqdm.tqdm(enumerate(self.templates)):

            for bias_type in self.bias_types:

                for group in bias_type.groups.values():

                    for def_word in group.definition_words:
                        
                        # Replace the <Group> token with current definition word
                        input = {k:v if k not in self.input_processor.input_names else self.replace_mask(v, self.group_token, def_word) for k, v in template.items()}

                        # Tokenize the input
                        input = self.input_processor.tokenize(input)

                        # Make tensors and batches of 1
                        input = {k: torch.tensor(v).unsqueeze(0).to(self.device) for k, v in input.items()}

                        # Find the position of the mask token
                        batch_index, token_index = self.get_mask_position(input["input_ids"])

                        # Use the model for predictions
                        output = self.model(**input)

                        # Extract the necessary score for the current definition word
                        target_term_probs, is_in_top_k = self.output_processor.process_output(output.logits, template[self.label_name], self.input_processor.tokenizer, batch_index, token_index, self.topk)

                        scores.append(
                            TaskOutput(
                                output=target_term_probs,
                                sentence_id=template_id,
                                def_word=def_word,
                                group=group.group_name,
                                bias_type=bias_type.bias_type_name,
                                gold_label=is_in_top_k
                            )
                        )
        
        return scores







class QuestionAnsweringTask(Task):

    def run(self):

        scores = []
        
        for template_id, template in tqdm.tqdm(enumerate(self.templates)):

            for bias_type in self.bias_types:

                for group in bias_type.groups.values():

                    for def_word in group.definition_words:
                        
                        # Replace the <Group> token with current definition word
                        input = {k:v if k not in self.input_processor.input_names else self.replace_mask(v, self.group_token, def_word) for k, v in template.items()}

                        # Tokenize the input
                        input = self.input_processor.tokenize(input)

                        # Make tensors and batches of 1
                        input = {k: torch.tensor(v).unsqueeze(0).to(self.device) for k, v in input.items()}

                        # Use the model for predictions
                        output = self.model(**input)

                        # Process the output
                        answer = self.output_processor.process_output(output, input)

                        scores.append(
                            TaskOutput(
                                output=answer,
                                sentence_id=template_id,
                                def_word=def_word,
                                group=group.group_name,
                                bias_type=bias_type.bias_type_name,
                                gold_label=template[self.label_name]
                            )
                        )
        
        return scores






















class ProbeForMaskedLanguageModeling(Task):

    def __init__(self, model, tokenizer, bias_types, templates, no_cuda=False):
        input_processor = SingleInputProcessor(tokenizer, truncation=True)
        output_processor = MaskedLanguageModelingOutputProcessor()
        template_transformer = MaskedLanguageModelingTemplateTransformer()
        super().__init__(model, bias_types, templates, input_processor, output_processor, template_transformer, no_cuda)
    


    def get_mask_position(self, input_ids):
        "Finds the position of the [MASK] in the input"
        mask_id = self.input_processor.tokenizer.convert_tokens_to_ids(self.input_processor.tokenizer.mask_token)
        batch_index, token_index = (input_ids == mask_id).nonzero(as_tuple=True)
        return batch_index.item(), token_index.item()


    def run(self):

        # Initialize the result dictionary
        result = {
            "per_template": []
        }

        # Transform the inputs to the format of the task
        self.transform_templates()

        for category, task_inputs in self.task_inputs.items():
            for i, task_input in enumerate(task_inputs):

                scores = {
                    "template": self.templates[category][i],
                    "category": category,
                    "scores": {}
                }

                for bias_type in self.bias_types:

                    scores["scores"][bias_type.bias_type_name] = {}

                    for group in bias_type.groups.values():

                        # Target terms are all definition words of the current group
                        target_terms = group.definition_words
                        
                        # Replace the <Group> token with mask token
                        task_input_for_current_group = [self.replace_mask(x, self.template_transformer.group_token, self.input_processor.tokenizer.mask_token) for x in task_input]

                        # Tokenize the input
                        input = self.input_processor.tokenize(task_input_for_current_group)

                        # Make tensors and batches of 1
                        input = {k: torch.tensor(v).unsqueeze(0).to(self.device) for k, v in input.items()}

                        # Find the position of the mask token
                        batch_index, token_index = self.get_mask_position(input["input_ids"])

                        # Use the model for predictions
                        logits = self.model(**input).logits

                        # Extract the necessary score for the current definition word
                        target_term_probs = self.output_processor.process_output(logits, target_terms, self.input_processor.tokenizer, batch_index, token_index)

                        # Take the mean of all defintion word subscores to get the score of the group
                        scores["scores"][bias_type.bias_type_name][group.group_name] = sum(target_term_probs) / len(target_term_probs)

                # Add all scores of the current template to the overall result
                result["per_template"].append(scores)

        return result
















class ProbeForSentimentAnalysis(Task):
    def __init__(self, model, tokenizer, bias_types, templates, output_dim=0, no_cuda=False):
        input_processor = SingleInputProcessor(tokenizer, truncation=True)
        output_processor = SoftmaxOutputProcessor(output_dim, softmax_dim=1)
        template_transformer = SingleInputSequenceClassificationTemplateTransformer()

        super().__init__(model, bias_types, templates, input_processor=input_processor, output_processor=output_processor, template_transformer=template_transformer, no_cuda=no_cuda)





class ProbeForTextualEntailment(Task):
    def __init__(self, model, tokenizer, bias_types, templates, output_dim=0, no_cuda=False):
        input_processor = DoubleInputProcessor(tokenizer, truncation=True)
        output_processor = SoftmaxOutputProcessor(output_dim, softmax_dim=1)
        template_transformer = DoubleInputSequenceClassificationTemplateTransformer()

        super().__init__(model, bias_types, templates, input_processor=input_processor, output_processor=output_processor, template_transformer=template_transformer, no_cuda=no_cuda)






class ProbeForParaphraseDetection(Task):
    def __init__(self, model, tokenizer, bias_types, templates, output_dim=0, no_cuda=False):
        input_processor = DoubleInputProcessor(tokenizer, truncation=True)
        output_processor = SoftmaxOutputProcessor(output_dim, softmax_dim=1)
        template_transformer = DoubleInputSequenceClassificationTemplateTransformer()

        super().__init__(model, bias_types, templates, input_processor=input_processor, output_processor=output_processor, template_transformer=template_transformer, no_cuda=no_cuda)










class ProbeForQuestionAnswering(Task):
    def __init__(self, model, tokenizer, bias_types, templates, no_cuda=False):
        input_processor = DoubleInputProcessor(tokenizer)
        output_processor = QuestionAsnweringOutputProcessor(tokenizer)
        template_transformer = QuestionAnsweringTemplateTransformer()

        super().__init__(model, bias_types, templates, input_processor, output_processor, template_transformer, no_cuda)

    
    def run(self):
        # Initialize the result dictionary
        result = {
            "per_template": []
        }

        # Transform the inputs to the format of the task
        self.transform_templates()

        for category, task_inputs in self.task_inputs.items():
            for i, task_input in enumerate(task_inputs):

                scores = {
                    "template": self.templates[category][i],
                    "category": category,
                    "scores": {}
                }

                for bias_type in self.bias_types:

                    scores["scores"][bias_type.bias_type_name] = {}

                    # Create an index from definition word to group
                    def_to_group = {}
                    for group in bias_type.groups.values():
                        scores["scores"][bias_type.bias_type_name][group.group_name] = []
                        for def_word in group.definition_words:
                            def_to_group[def_word] = group.group_name
                    

                    # Find pairings of definition words across groups
                    paired_definitions = zip_longest_with_cycling(*[g.definition_words for g in bias_type.groups.values()])
                    
                    for pairing in paired_definitions:
                        
                        # Replace the <Group> token with group mentions
                        task_input_for_current_pairing = [self.replace_mask(x, self.template_transformer.group_token, ", ".join(pairing)) for x in task_input]
                        
                        # Tokenize the input
                        input = self.input_processor.tokenize(task_input_for_current_pairing)

                        # Make tensors and batches of 1
                        input_tensor = {k: torch.tensor(v).unsqueeze(0).to(self.device) for k, v in input.items()}

                        # Use the model for predictions
                        output = self.model(**input_tensor)

                        # Extract the scores for the current pairing
                        sub_scores = self.output_processor.process_output(output, input, pairing)
                        for def_word, sub_score in sub_scores.items():
                            scores["scores"][bias_type.bias_type_name][def_to_group[def_word]].append(sub_score)


                    # Take the mean of all defintion word subscores to get the score of each group
                    for g, group_sub_scores in scores["scores"][bias_type.bias_type_name].items():
                        scores["scores"][bias_type.bias_type_name][g] = sum(group_sub_scores) / len(group_sub_scores)

                # Add all scores of the current template to the overall result
                result["per_template"].append(scores)

        return result














class ProbeForMultipleChoice(Task):

    def __init__(self, model, tokenizer, bias_types, templates, no_cuda=False):
        input_processor = MultipleChoiceInputProcessor(tokenizer)
        output_processor = MultipleChoiceOutputProcessor()
        template_transformer = MultipleChoiceTemplateTransformer()

        super().__init__(model, bias_types, templates, input_processor, output_processor, template_transformer, no_cuda)
    





    def run(self):
        # Initialize the result dictionary
        result = {
            "per_template": []
        }

        # Transform the inputs to the format of the task
        self.transform_templates()

        for category, task_inputs in self.task_inputs.items():
            for i, task_input in enumerate(task_inputs):

                scores = {
                    "template": self.templates[category][i],
                    "category": category,
                    "scores": {}
                }

                for bias_type in self.bias_types:

                    scores["scores"][bias_type.bias_type_name] = {}

                    # Create an index from definition word to group
                    def_to_group = {}
                    for group in bias_type.groups.values():
                        scores["scores"][bias_type.bias_type_name][group.group_name] = []
                        for def_word in group.definition_words:
                            def_to_group[def_word] = group.group_name
                    

                    # Find pairings of definition words across groups
                    paired_definitions = zip_longest_with_cycling(*[g.definition_words for g in bias_type.groups.values()])
                    
                    for pairing in paired_definitions:
                        
                        # Replace the <Group> token with group mentions
                        task_input_for_current_pairing = (
                            task_input[0],
                            [self.replace_mask(task_input[1], self.template_transformer.group_token, p) for p in pairing]
                        )

                        # Tokenize the input
                        input = self.input_processor.tokenize(task_input_for_current_pairing)

                        # Make tensors and batches
                        input = {k: torch.tensor(v).unsqueeze(0).to(self.device) for k, v in input.items()}
                        
                        # Transpose to make the shape (num_choices, batch_size) because the multiple choice head expects this shape as input
                        for k, _ in input.items():
                            input[k] = torch.transpose(input[k], 0, 1)

                        # Use the model for predictions
                        output = self.model(**input)

                        # Extract the scores for the current pairing
                        sub_scores = self.output_processor.process_output(output)
                        for i, sub_score in enumerate(sub_scores):
                            scores["scores"][bias_type.bias_type_name][def_to_group[pairing[i]]].append(sub_score)


                    # Take the mean of all defintion word subscores to get the score of each group
                    for g, group_sub_scores in scores["scores"][bias_type.bias_type_name].items():
                        scores["scores"][bias_type.bias_type_name][g] = sum(group_sub_scores) / len(group_sub_scores)

                # Add all scores of the current template to the overall result
                result["per_template"].append(scores)

        return result







    