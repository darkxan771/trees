# The generating series of the various combinatorial classes

import numpy as np


def generating_series_SP(N: int) -> list[int]:
    """
    Computes the N first terms of the generating series of
    set partitions (Bell numbers).
    """
    res = [1, 1]
    k = 1
    row = np.array([1])
    while k < N:
        new_row = np.zeros(k + 1, dtype=int)
        new_row[0] = row[-1]
        for i in range(1, k + 1):
            new_row[i] = row[i - 1] + new_row[i - 1]
        row = new_row
        k += 1
        res.append(int(row[-1]))
    return res


def generating_series_P(N: int) -> list[int]:
    """
    Computes the N first terms of the generating series of
    integer partitions.
    """
    divs = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            divs[j].append(i)
    res = [0] * (N + 1)
    sigma = [sum(x, 0) for x in divs]
    res[0] = 1
    for n in range(1, N + 1):
        res[n] = sum([sigma[n - k] * res[k] for k in range(n)]) // n
    return res


def generating_series_RT(N: int) -> list[int]:
    """
    Computes the N first terms of the generating series of
    recursive trees.
    """
    res = [0] * (N + 1)
    res[1] = 1
    for n in range(1, N):
        res[n + 1] = res[n] * n
    return res


def generating_series_DRT(N: int) -> list[int]:
    """
    Computes the N first terms of the generating series of
    double recursive trees.
    """
    res = [0] * (N + 1)
    res[1] = 1
    for n in range(1, N):
        res[n + 1] = res[n] * int(n * (n + 1) / 2)
    return res


def generating_series_T(N: int) -> list[int]:
    """
    Computes the N first terms of the generating series of
    rooted unlabelled trees.
    """
    divs = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            divs[j].append(i)
    res = [0] * (N + 1)
    res2 = [0] * (N + 1)
    res[1] = 1
    res2[1] = 1
    for n in range(1, N):
        res[n + 1] = sum([res[n - k] * res2[k + 1] for k in range(n)]) // n
        res2[n + 1] = sum([d * res[d] for d in divs[n + 1]])
    return res
