import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from src.metrics.distances import singleton_data_type, set_data_type

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