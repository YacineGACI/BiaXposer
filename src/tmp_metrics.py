class BiasMetric:
    def __call__(self, scores):
        raise NotImplementedError




class AbsoluteDivergenceFromExpectedOutcome(BiasMetric):
    def __call__(self, scores):
        scores = [s/sum(scores) for s in scores]
        expected = 1/len(scores)
        return sum([abs(s - expected) for s in scores]) / 2 # Division by 2 such that bias would mean the percentage of space that is wrongly distributed 




class NormalizedAbsoluteDivergenceFromExpectedOutcome(BiasMetric):
    """
        Like AbsoluteDivergenceFromExpectedOutcome, but the bias scores are bound between 0% and 100%
    """
    def __call__(self, scores):
        scores = [s/sum(scores) for s in scores]
        expected = 1/len(scores)
        score = sum([abs(s - expected) for s in scores]) / 2
        maximum_possible_score = 1 - 1 / len(scores)
        return score / maximum_possible_score