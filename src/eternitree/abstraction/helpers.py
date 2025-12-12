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


def shift_array(L: np.ndarray, l: int, n: int) -> None:
    """
    Shift by one index all the entries of an array
    between l and n (assuming that said array has
    at least n+1 entries).
    """
    L[l + 1 : n + 1] = L[l:n]


def shift_dict(d: dict, l: int, n: int, sign: int = 1) -> dict:
    """
    Shift by one index all the keys of a dictionary between l and n.
    Put sign = -1 to shift in the other direction (then, the (l-1)-th
    entry is deleted).
    """
    res = {}
    L = list(d.keys())
    L.sort()
    for k in L:
        if l <= k < n:
            res[k + sign] = d[k]
        else:
            res[k] = d[k]
    return res
