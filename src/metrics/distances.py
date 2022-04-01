from scipy.stats import wasserstein_distance as wd

def absolute_distance(x, y):
    return abs(x - y)


def wasserstein_distance(x, y):
    return wd(x, y)


def standard_deviation_distance(x, y):
    pass
