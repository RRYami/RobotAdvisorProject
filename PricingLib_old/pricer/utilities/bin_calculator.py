# Freedman–Diaconis rule

import numpy as np
import math


def bin_calculator(data):
    """
    Calculate the number of bins for a histogram using the Freedman–Diaconis rule.
    """
    data = np.array(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    h = 2 * iqr * math.pow(len(data), -1 / 3)
    bins = math.ceil((max(data) - min(data)) / h)
    return bins
