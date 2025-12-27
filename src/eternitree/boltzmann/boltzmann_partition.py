# Constructs a Boltzmann sampler for uniform random partitions

import numpy as np
import scipy.stats as scs
from scipy.optimize import brentq

from ..abstraction import IntegerPartition


def expectation_size(x: float) -> float:
    """
    Computes the expected size of the random partition
    with Boltzmann parameter x.
    """
    s = 0.0
    k = 1
    while True:
        term = k * (x**k) / (1 - x**k)
        s += term
        if term < 1e-50:
            break
        k += 1
    return float(s)


def find_x_for_n(n: int) -> float:
    """
    Finds the Boltzmann parameter x in order to obtain
    a partition with expected size n.
    """

    def f(x: float):
        return float(expectation_size(x) - n)

    res, _ = brentq(f, 0.3333, 0.9999, full_output=True)
    return float(res)


def boltzmann_sampler(x: float) -> IntegerPartition:
    """
    Picks at random an integer partition.

    Each integer partition L has probability x^{|L|} / P(x), where P(x)
    is the generating series of the class of integer partitions.
    """
    kmax = int(np.ceil(np.log(1e-50) / np.log(x)))
    mult: dict[int, int] = {}
    for k in range(1, kmax):
        m = int(scs.geom(p=1 - x**k).rvs() - 1)
        if m > 0:
            mult[k] = m
    res = sum(([k] * m for (k, m) in mult.items()), [])
    res.reverse()
    return IntegerPartition(res)
