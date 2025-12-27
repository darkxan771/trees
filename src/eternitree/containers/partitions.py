# Defines an IntegerPartitions container

from typing import Iterator

from ..abstraction import CombinatorialClass
from ..abstraction import IntegerPartition
from .generating_series import generating_series_P


class _IntegerPartitionsIterator_n(Iterator):
    def __init__(self, n: int):
        self.a = [0] * (n + 1)
        self.k = 1
        self.x = 1
        self.y = n - 1
        self.end_of_cycle = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.k == 0 and self.end_of_cycle:
            raise StopIteration
        else:
            if self.end_of_cycle:
                self.end_of_cycle = False
                self.x = self.a[self.k - 1] + 1
                self.k -= 1
            while 2 * self.x <= self.y:
                self.a[self.k] = self.x
                self.y -= self.x
                self.k += 1
            if self.x <= self.y:
                self.a[self.k] = self.x
                self.a[self.k + 1] = self.y
                res = IntegerPartition(self.a[self.k + 1 :: -1])
                self.x += 1
                self.y -= 1
                return res
            else:
                self.a[self.k] = self.x + self.y
                self.y = self.x + self.y - 1
                res = IntegerPartition(self.a[self.k :: -1])
                self.end_of_cycle = True
                return res


class IntegerPartitions(CombinatorialClass):
    """
    A container for integer partitions.

    If a size n is given, the container is restricted to integer
    partitions with size n. In any case, one can iterate upon the
    container.
    """

    def __init__(self, n: int | None = None):
        self.category = "partition"
        self.order = n

    @classmethod
    def generating_series(cls, N: int) -> list[int]:
        return generating_series_P(N)

    @classmethod
    def iter_n(cls):
        return lambda n: _IntegerPartitionsIterator_n(n)

    @staticmethod
    def example() -> IntegerPartition:
        """
        An example of integer partition.
        """
        return IntegerPartition([5, 3, 2, 2, 1, 1, 1])
