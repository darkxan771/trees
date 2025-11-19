import numpy as np
import numpy.random as rand

from collections.abc import Sequence


def _c_flatten(data):
    for x in data:
        yield from x


def nested_list_to_code(L: list) -> tuple:
    """
    Converts a nested list into the code of the corresponding rooted tree.

    The result is standardised.
    """
    C = [tuple(x + 1 for x in nested_list_to_code(c)) for c in L]
    C.sort(reverse=True)
    return tuple(_c_flatten([(0,)] + C))


def code_to_nested_list(L: Sequence[int]) -> list:
    """
    Converts the code of a rooted tree into the corresponding nested list.

    The result is standardised.
    """
    children = []
    if not (L[0] == 0):
        raise ValueError
    if len(L) > 1:
        child = [0]
        for i in range(2, len(L)):
            if L[i] == 1:
                children.append(child)
                child = [0]
            else:
                child.append(L[i] - 1)
        children.append(child)
    children.sort(key=(lambda x: tuple(x)), reverse=True)
    return [code_to_nested_list(child) for child in children]


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


def permutation_to_code(p: np.ndarray) -> np.ndarray:
    """
    Converts a permutation array to the code array of a recursive tree.
    """
    res = np.zeros(p.size, dtype=int)
    for i in range(1, len(p)):
        res[i] = np.count_nonzero(p[:i] < p[i])
    return res


def code_to_permutation(c: np.ndarray) -> np.ndarray:
    """
    Converts the code array of a recursive tree to a permutation array.
    """
    res = np.zeros(c.size, dtype=int)
    for i in range(1, len(c)):
        res[i] = c[i]
        res[:i] += res[:i] >= res[i]
    return res
