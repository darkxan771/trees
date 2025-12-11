# Defines an IntegerPartitions container

from itertools import chain, count

from ..abstraction import InfiniteSetError, IntegerPartition
from ..boltzmann.boltzmann_partition import generating_series_P


class _IntegerPartitionsIterator_n:
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


class IntegerPartitions:
    """
    A container for integer partitions with a given size n.
    """

    def __init__(self, n: None | int = None):
        self.order = n

    def __repr__(self):
        if self.order is None:
            return "Integer partitions"
        else:
            return f"Integer partitions with size {self.order}"

    def __contains__(self, L):
        A = isinstance(L, IntegerPartition)
        if self.order is not None:
            return A and L.size == self.order
        else:
            return A

    def __iter__(self):
        if self.order is not None:
            return _IntegerPartitionsIterator_n(self.order)
        else:
            return chain.from_iterable(
                IntegerPartitions(n) for n in count(1)
            ).__iter__()

    def cardinality(self) -> int:
        """
        Returns the cardinality of the set of integer partitions with
        size n.

        The cardinality satisfies the recurrence relation:

        P[n]
        = 1/n sum([d * P[n-k] for k in 1..n and for d | k]).
        """
        if self.order is None:
            raise InfiniteSetError("Infinite set")
        else:
            return int(generating_series_P(self.order)[-1])
