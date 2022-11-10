import re, string, collections
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from biaxposer.metrics.distances import singleton_data_type, set_data_type

class ScoringFunction:
    def __init__(self, name=None, data_type=None):
        self.name = name
        self.data_type = data_type

    def __call__(self, predictions, labels):
        raise NotImplementedError



class F1_Score(ScoringFunction):
    def __init__(self, average="binary"):
        self.average = average
        super().__init__("F1 score", singleton_data_type)

    def __call__(self, predictions, labels):
        return f1_score(labels, [np.argmax(p) for p in predictions], average=self.average)



class Accuracy_Score(ScoringFunction):
    def __init__(self):
        super().__init__("Accuracy", singleton_data_type)

    def __call__(self, predictions, labels):
        return accuracy_score(labels, [np.argmax(p) for p in predictions])



class Precision_Score(ScoringFunction):
    def __init__(self, average="binary"):
        self.average = average
        super().__init__("Precision", singleton_data_type)

    def __call__(self, predictions, labels):
        return precision_score(labels, [np.argmax(p) for p in predictions], average=self.average)



class Recall_Score(ScoringFunction):
    def __init__(self, average="binary"):
        self.average = average
        super().__init__("Recall", singleton_data_type)
        
    def __call__(self, predictions, labels):
        return recall_score(labels, [np.argmax(p) for p in predictions], average=self.average)



class ClassPrediction_Score(ScoringFunction):
    def __init__(self, class_id=None):
        self.class_id = class_id
        super().__init__("Class Prediction", set_data_type)

    def __call__(self, predictions, labels=None):
        res = []
        for i, p in enumerate(predictions):
            class_id = self.class_id if self.class_id is not None else labels[i]
            res.append(p[class_id])
        return res




class AverageClassPrediction_Score(ScoringFunction):
    def __init__(self, class_id=None):
        self.class_id = class_id
        super().__init__("Average Class Prediction", singleton_data_type)

    def __call__(self, predictions, labels=None):
        res = []
        for i, p in enumerate(predictions):
            class_id = self.class_id if self.class_id is not None else labels[i]
            res.append(p[class_id])
        return sum(res) / len(res)





class Likelihood_Score(ScoringFunction):
    def __init__(self, word_id=None):
        self.word_id = word_id
        super().__init__("Likelihood", set_data_type)

    def __call__(self, predictions, labels=None):
        res = []
        for p in predictions:
            if self.word_id is not None:
                res.append(p[self.word_id])
            else:
                res.append(sum(p) / len(p))
        return res




class AverageLikelihood_Score(ScoringFunction):
    def __init__(self, word_id=None):
        self.word_id = word_id
        super().__init__("Average Likelihood", singleton_data_type)

    def __call__(self, predictions, labels=None):
        res = []
        for p in predictions:
            if self.word_id is not None:
                res.append(p[self.word_id])
            else:
                res.append(sum(p) / len(p))
        return sum(res) / len(res)




class AccuracyForLM_Score(ScoringFunction):
    def __init__(self):
        super().__init__("Accuracy for LM", singleton_data_type)

    
    def __call__(self, predictions, labels):
        accuracy = 0
        for l in labels:
            if True in l:
                accuracy += 1
        return accuracy / len(labels)



    


class QA_Score(ScoringFunction):
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def get_tokens(self, s):
        if not s: return []
        return self.normalize_answer(s).split()


    def compute_exact(self, a_gold, a_pred):
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))


    def compute_f1(self, a_gold, a_pred):
        gold_toks = self.get_tokens(a_gold)
        pred_toks = self.get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1




class F1ForQA_Score(QA_Score):
    def __init__(self):
        super().__init__("F1 for QA", set_data_type)
    
    def __call__(self, predictions, labels):
        res = []
        for i in range(len(predictions)):
            res.append(self.compute_f1(labels[i], predictions[i]))
        return res



class AverageF1ForQA_Score(QA_Score):
    def __init__(self):
        super().__init__("Average F1 for QA", singleton_data_type)
    
    def __call__(self, predictions, labels):
        res = []
        for i in range(len(predictions)):
            res.append(self.compute_f1(labels[i], predictions[i]))
        return sum(res) / len(res)



class ExactMatchForQA_Score(QA_Score):
    def __init__(self):
        super().__init__("EM for QA", set_data_type)
    
    def __call__(self, predictions, labels):
        res = []
        for i in range(len(predictions)):
            res.append(self.compute_exact(labels[i], predictions[i]))
        return res



class AverageExactMatchForQA_Score(QA_Score):
    def __init__(self):
        super().__init__("Average EM for QA", singleton_data_type)
    
    def __call__(self, predictions, labels):
        res = []
        for i in range(len(predictions)):
            res.append(self.compute_exact(labels[i], predictions[i]))
        return sum(res) / len(res)



########################################################################
####    Scoring functions to be used for the failure rate metric    ####
########################################################################


class FailureRateSFForSequenceClassification(ScoringFunction):
    def __init__(self):
        super().__init__("Scoring fucntion for Failure Rate for Sequence Classification", singleton_data_type)

    def __call__(self, predictions, labels):
        mean_predictions_across_def_words = np.mean(predictions, 0)
        label = labels[0] # Since all labels for def words must be the same
        return mean_predictions_across_def_words[label]



class FailureRateSFForLanguageModeling(ScoringFunction):
    def __init__(self):
        super().__init__("Scoring fucntion for Failure Rate for Language Modeling", singleton_data_type)

    def __call__(self, predictions, labels):
        return np.mean(predictions) # Mean of all target words across all def words



class FailureRateSFForQuestionAnswering(AverageF1ForQA_Score):
    def __init__(self):
        super().__init__()
