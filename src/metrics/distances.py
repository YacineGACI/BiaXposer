from scipy.stats import wasserstein_distance as wd

def absolute_distance(x, y):
    return abs(x - y)


def wasserstein_distance(x, y):
    return wd(x, y)


def absolute_divergence_from_expected_outcome(x):
    scores = [s/sum(x) for s in x]
    expected = 1/len(scores)
    return sum([abs(s - expected) for s in scores]) / 2 # Division by 2 such that bias would mean the percentage of space that is wrongly distributed




def normlaized_absolute_divergence_from_expected_outcome(x):
    scores = [s/sum(x) for s in x]
    expected = 1/len(scores)
    score = sum([abs(s - expected) for s in scores]) / 2
    maximum_possible_score = 1 - 1 / len(scores)
    return score / maximum_possible_score