# Useful functions for the computation of probabilities of random partitions

from typing import Callable

import numpy as np
from scipy.special import factorial

from ..abstraction import IntegerPartition
from ..containers import IntegerPartitions


def probability_uniform(L: IntegerPartition) -> float:
    """
    Returns the probability of L under the Plancherel measure.
    """
    return float(1 / IntegerPartitions(L.size).cardinality)


def probability_plancherel(L: IntegerPartition) -> float:
    """
    Returns the probability of L under the Plancherel measure.
    """
    return float(L.dimension**2 / factorial(L.size, exact=True))


def probability_ewens(L: IntegerPartition, theta: float) -> float:
    """
    Returns the probability of L under the Ewens measure with
    parameter theta.
    """
    n = L.size
    A = np.prod((1 + np.arange(n)) / (theta + np.arange(n)))
    B = theta**L.length / L.z
    return float(A * B)


compute_probabilities: dict[str, Callable] = {
    "plancherel": lambda L, _: probability_plancherel(L),
    "ewens": probability_ewens,
    "uniform": lambda L, _: probability_uniform(L),
}
