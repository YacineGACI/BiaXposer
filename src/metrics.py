class BiasMetric:
    def __call__(self, scores):
        raise NotImplementedError




class AbsoluteDivergenceFromExpectedOutcome(BiasMetric):
    def __call__(self, scores):
        scores = [s/sum(scores) for s in scores]
        expected = 1/len(scores)
        return sum([abs(s - expected) for s in scores])