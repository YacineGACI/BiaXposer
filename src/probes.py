import torch

class Probe:
    def __init__(self, model, bias_types, templates, input_processor=None, output_processor=None, template_transformer=None, no_cuda=False):
        self.model = model
        self.bias_types = bias_types
        self.templates = templates
        self.input_processor = input_processor
        self.output_processor = output_processor
        self.template_transformer = template_transformer
        self.device = 'cuda' if not no_cuda and torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)


    
    def replace_mask(self, template, mask, replacement):
        return template.replace(mask, replacement)

    

    def transform_templates(self):
        """
            Takes the task-independent templates and transforms them according to the task's input format
        """
        self.task_inputs = {
            category: [
                self.template_transformer.transform(t) for t in templates
            ] for category, templates in self.templates.items()
        }



    def run(self):
        raise NotImplementedError

















class ProbeForSequenceClassification(Probe):
    def __init__(self, model, bias_types, templates, input_processor=None, output_processor=None, template_transformer=None, device=False):
        super().__init__(model, bias_types, templates, input_processor, output_processor, template_transformer, device)




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
                        
                        # Initialize an empty list to store outputs corresponding to each definition word of this group
                        subscores_for_definition_words = []
                        for def_word in group.definition_words:
                            
                            # Replace the <Group> token with current definition word
                            task_input_for_current_group = [self.replace_mask(x, self.template_transformer.group_token, def_word) for x in task_input]

                            # Tokenize the input
                            input = self.input_processor.tokenize(task_input_for_current_group)

                            # Make tensors and batches of 1
                            input = {k: torch.tensor(v).unsqueeze(0).to(self.device) for k, v in input.items()}

                            # Use the model for predictions
                            logits = self.model(**input).logits

                            # Extract the necessary score for the current definition word
                            sub_score = self.output_processor.process_output(logits)

                            subscores_for_definition_words.append(sub_score)

                        # Take the mean of all defintion word subscores to get the score of the group
                        scores["scores"][bias_type.bias_type_name][group.group_name] = sum(subscores_for_definition_words) / len(subscores_for_definition_words)

                # Add all scores of the current template to the overall result
                result["per_template"].append(scores)

        return result

