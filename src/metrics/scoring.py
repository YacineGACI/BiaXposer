import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class ScoringFunction:
    def __call__(self, predictions, labels):
        raise NotImplementedError



class F1_Score(ScoringFunction):
    def __call__(self, predictions, labels):
        return f1_score(labels, [np.argmax(p) for p in predictions])



class Accuracy_Score(ScoringFunction):
    def __call__(self, predictions, labels):
        return accuracy_score(labels, [np.argmax(p) for p in predictions])



class Precision_Score(ScoringFunction):
    def __call__(self, predictions, labels):
        return precision_score(labels, [np.argmax(p) for p in predictions])



class Recall_Score(ScoringFunction):
    def __call__(self, predictions, labels):
        return recall_score(labels, [np.argmax(p) for p in predictions])



class ClassPrediction_Score(ScoringFunction):
    def __init__(self, class_id=None):
        self.class_id = class_id

    def __call__(self, predictions, labels=None):
        # print(predictions)
        res = []
        for i, p in enumerate(predictions):
            class_id = self.class_id if self.class_id is not None else labels[i]
            res.append(p[class_id])
        return res