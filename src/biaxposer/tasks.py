import tqdm

import torch



class TaskOutput:
    def __init__(self, output, sentence_id, def_word, group, bias_type, gold_label, template_id):
        self.output = output
        self.sentence_id = sentence_id
        self.def_word = def_word
        self.group = group
        self.bias_type = bias_type
        self.gold_label = gold_label
        self.template_id = template_id


    def __str__(self):
        return '''
            output: {},
            id: {},
            def_word: {},
            group: {},
            bias_type: {},
            gold_label: {},
            template_id: {}
        
        '''.format(self.output, self.sentence_id, self.def_word, self.group, self.bias_type, self.gold_label, self.template_id)


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
                                gold_label=template[self.label_name],
                                template_id=template["t_id"]
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
                                gold_label=is_in_top_k,
                                template_id=template["t_id"]
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
                                gold_label=template[self.label_name],
                                template_id=template["t_id"]
                            )
                        )
        
        return scores
