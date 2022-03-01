import torch
from src.utils import find_span_edges

class OutputProcessor:
    def __init__(self):
        pass

    def process_output(self, output):
        raise NotImplementedError





class SingleOutputProcessor(OutputProcessor):

    def process_output(self, output):
        return output[0, 0].item()





class SoftmaxOutputProcessor(OutputProcessor):
    def __init__(self, output_dim, softmax_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.softmax_dim = softmax_dim

    def process_output(self, output):
        logits = output
        probabilities = torch.softmax(logits, self.softmax_dim)
        return probabilities[0, self.output_dim].item() # 0 for first and only batch




class MaskedLanguageModelingOutputProcessor(OutputProcessor):

    def process_output(self, output, target_terms, tokenizer, batch_index, token_index):
        probs = torch.softmax(output[batch_index, token_index], 0)
        res = []
        for t in target_terms:
            res.append(probs[tokenizer.convert_tokens_to_ids(t)].item())
        return res
            



class QuestionAsnweringOutputProcessor(OutputProcessor):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        super().__init__()



    def process_output(self, output, input, target_terms):
        start_logits = output.start_logits
        end_logits = output.end_logits

        start_probs = torch.softmax(start_logits, dim=1)
        end_probs = torch.softmax(end_logits, dim=1)

        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)

        res = {}
        for t in target_terms:
            current_term_tokens = self.tokenizer(" " + t) # Plus space because Roberta changes the token id when it is first in in the input 
            term_start_position, term_end_position = find_span_edges(input['input_ids'], current_term_tokens['input_ids'], sep_token_id)
            this_term_score = start_probs[0, term_start_position] + end_probs[0, term_end_position]
            res[t] = this_term_score.item()

        return res


