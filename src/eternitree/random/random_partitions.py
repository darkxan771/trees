# Defines abstract classes RandomPartition and RandomSetPartition

from collections import defaultdict

from ..abstraction import IntegerPartition
from ..containers import IntegerPartitions
from .partition_generators import generate_random
from .partition_probabilities import compute_probabilities


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
