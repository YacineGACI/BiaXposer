from scipy.stats import wasserstein_distance as wd

singleton_data_type = "singleton"
set_data_type = "set"


class Distance:
    def __init__(self, name=None, data_type=None, eval_type=None):
        self.name = name
        self.data_type = data_type
        self.eval_type = eval_type





class AbsoluteDistance(Distance):
    def __init__(self):
        super().__init__("Absolute Distance", singleton_data_type, ["pcm", "bcm"])

    def __call__(self, x, y):
        return abs(x - y)





class WassersteinDistance(Distance):
    def __init__(self):
        super().__init__("Wasserstein Distance", set_data_type, ["pcm", "bcm"])

    def __call__(self, x, y):
        return wd(x, y)
    




class AbsoluteDivergenceFromExpectedOutcome(Distance):
    def __init__(self, normalized=False):
        super().__init__("Absolute Divergence From Expected Outcome", singleton_data_type, ["mcm"])
        self.normalized = normalized

    def __call__(self, x):
        scores = [s/sum(x) for s in x]
        expected = 1/len(scores)
        score = sum([abs(s - expected) for s in scores]) / 2 # Division by 2 such that bias would mean the percentage of space that is wrongly distributed
        if self.normalized:
            maximum_possible_score = 1 - 1 / len(scores)
            return score / maximum_possible_score
        else:
            return score
