import numpy as np
import scipy.stats as stats
a = np.array([11.0, 37, 35, 25, 22, 17, 87, 91, 9, 49, 38, 81, 86, 58, 22, 90, 43, 1, 74, 66])

# Print the geometric mean of the array
print(np.std(a))

# print kurtosis
print(stats.kurtosis(a, fisher=False, bias=False))
print(stats.skew(a))
