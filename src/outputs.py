import torch

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
            

