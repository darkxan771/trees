# Defines abstract classes RandomPartition and RandomSetPartition


from ..abstraction.random import Random
from ..containers import IntegerPartitions
from ..containers import SetPartitions


class RandomSetPartition(Random):
    """
    A distribution of random set partitions of a given set.
    """

    def __init__(
        self, L: int | list[int], name: str, parameter: float | None = None
    ):
        if isinstance(L, int):
            self.set = list(range(L))
            self.size = L
        else:
            self.set = L
            self.set.sort()
            self.size = len(L)
        self.object = "set partition"
        self.name = name
        self.parameter = parameter

    @property
    def label(self):
        return self.set

    def container(self):
        return SetPartitions(self.set)


class RandomPartition(Random):
    """
    A distribution of integer partitions with a given size n.
    """

    def __init__(self, n: int, name: str, parameter: float | None = None):
        self.size = n
        self.object = "partition"
        self.name = name
        self.parameter = parameter

    @property
    def label(self):
        return self.size

    def container(self):
        return IntegerPartitions(self.size)


def UniformSetPartition(L: int | list[int]) -> RandomSetPartition:
    return RandomSetPartition(L, "uniform")


def UniformPartition(n: int) -> RandomPartition:
    return RandomPartition(n, "uniform")


def PlancherelPartition(n: int) -> RandomPartition:
    return RandomPartition(n, "plancherel")


def EwensSetPartition(
    L: int | list[int], theta: float = 1
) -> RandomSetPartition:
    return RandomSetPartition(L, "ewens", theta)


def EwensPartition(n: int, theta: float = 1) -> RandomPartition:
    return RandomPartition(n, "ewens", theta)
