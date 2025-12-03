# Useful functions and classes

import numpy as np


class InfiniteSetError(Exception):
    pass


def standardisation(L: np.ndarray) -> np.ndarray:
    """
    Converts an array of distinct numbers into the corresponding
    permutation.
    """
    n = L.size
    res = np.ones(n, dtype=int)
    for k in range(1, n):
        for j in range(k):
            res[j] += int(L[j] > L[k])
            res[k] += int(L[j] < L[k])
    return res
