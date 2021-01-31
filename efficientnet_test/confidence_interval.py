import numpy as np


def get_z(confidence_level=95):
    """
    Compute the z
    :param confidence_level:
    :return:
    """
    if confidence_level == 99:
        return 2.576
    elif confidence_level == 98:
        return 2.326
    elif confidence_level == 95:
        return 1.96
    elif confidence_level == 90:
        return 1.645


def confidence_interval(x, confidence_level=95):
    """
    Computes the confidence interval using t-distribution
    :param x: an array numbers for which to compute the confidence interval
    :param confidence_level: the level of confidence (90, 95, 98 or 99)
    :return: a tuple representing the interval (limits of the interval)
    """
    z = get_z(confidence_level)
    n_root = len(x)
    x_mean = np.mean(x)
    s = np.std(x)
    margin_error = z * s / n_root
    return x_mean - margin_error, x_mean + margin_error
