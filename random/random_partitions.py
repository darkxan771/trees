# Defines an abstract class RandomPartition, and various subclasses

import numpy as np
import numpy.random as rand

from collections import defaultdict
from scipy.special import factorial
from typing import Sequence, Callable
from ..abstraction.helpers import standardisation
from ..abstraction import IntegerPartition
from ..containers import IntegerPartitions
from ..boltzmann.boltzmann_partition import find_x_for_n, boltzmann_sampler


def probability_uniform(L: IntegerPartition) -> float:
    """
    Returns the probability of L under the Plancherel measure.
    """
    return float(1 / IntegerPartitions(L.size).cardinality())


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


def _RSK_P(w: Sequence[int]) -> list:
    def insert_in_row(x: int, R: list[int]) -> tuple[list[int], None | int]:
        if all(x >= r for r in R):
            return R + [x], None
        else:
            i = R.index([r for r in R if r > x][0])
            return R[:i] + [x] + R[i + 1 :], R[i]

    res = [[w[0]]]
    for x in w[1:]:
        IR = insert_in_row(x, res[0])
        res[0] = IR[0]
        k = 1
        while IR[1] is not None:
            if k + 1 > len(res):
                res.append([IR[1]])
                IR = [], None
            else:
                IR = insert_in_row(IR[1], res[k])
                res[k] = IR[0]
                k += 1
    return res


def generate_uniform(n: int) -> IntegerPartition:
    """
    Generates a random integer partition with size n and
    uniform distribution.
    """
    x = find_x_for_n(n)
    res = boltzmann_sampler(x)
    while res.size != n:
        res = boltzmann_sampler(x)
    return res


def generate_plancherel(n: int) -> IntegerPartition:
    """
    Generates a random integer partition with size n and
    Plancherel distribution.
    """
    U = rand.random(size=n)
    perm = standardisation(U).tolist()
    return IntegerPartition([len(R) for R in _RSK_P(perm)])


def generate_ewens(n: int, theta: float) -> IntegerPartition:
    """
    Generates a random integer partition with size n and
    Ewens distribution with parameter theta.
    """
    res = []
    for k in range(n):
        test = rand.random() <= theta / (theta + k)
        if test:
            res.append(1)
        else:
            ind = rand.choice(len(res), p=np.array(res) / k)
            res[ind] += 1
    res.sort(reverse=True)
    return IntegerPartition(res)


compute_probabilities: dict[str, Callable] = {
    "plancherel": (lambda L, T: probability_plancherel(L)),
    "ewens": probability_ewens,
    "uniform": (lambda L, T: probability_uniform(L)),
}


generate_random: dict[str, Callable] = {
    "plancherel": (lambda n, T: generate_plancherel(n)),
    "ewens": generate_ewens,
    "uniform": (lambda n, T: generate_uniform(n)),
}


class RandomPartition:
    def __init__(self, n: int, name: str, parameter: float | None = None):
        self.size = n
        self.name = name
        self.parameter = parameter

    def __repr__(self):
        res = f"Random partition with size {self.size}"
        res += f" and {self.name.capitalize()} distribution"
        if self.parameter is not None:
            res += f"(parameter = {self.parameter})"
        return res

    def probability(self, L: IntegerPartition) -> float:
        """
        Computes the probability of L under the prescribed
        distribution.
        """
        if L.size == self.size:
            return compute_probabilities[self.name](L, self.parameter)
        else:
            return float(0)

    def distribution(self) -> defaultdict:
        """
        Returns a dictionary with items (L, probability[L]), where L runs
        over the set of integer partitions with size n, and probability[L]
        is the probability of L under the prescribed distribution.
        """
        return defaultdict(
            float,
            {
                tuple(L.parts): self.probability(L)
                for L in IntegerPartitions(self.size)
            },
        )

    def get_random_element(self) -> IntegerPartition:
        """
        Picks at random an integer partition under the prescribed
        distribution.
        """
        return generate_random[self.name](self.size, self.parameter)


def UniformPartition(n: int):
    return RandomPartition(n, "uniform")


def PlancherelPartition(n: int):
    return RandomPartition(n, "plancherel")


def EwensPartition(n: int, theta: float = 1):
    return RandomPartition(n, "ewens", theta)
