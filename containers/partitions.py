from ..abstraction.partition import IntegerPartition
from ..random.boltzmann_partition import generating_series_P


class _IntegerPartitionsIterator_n:
    def __init__(self, n: int):
        self.size = n
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

    def __init__(self, n: int):
        self.size = n

    def __repr__(self):
        return f"Integer partitions with size {self.size}"

    def __iter__(self):
        return _IntegerPartitionsIterator_n(self.size)

    def __contains__(self, L):
        A = isinstance(L, IntegerPartition)
        return A and L.size == self.size

    def cardinality(self) -> int:
        """
        Returns the cardinality of the set of integer partitions with
        size n.

        The cardinality satisfies the recurrence relation:

        P[n]
        = 1/n sum([d * P[n-k] for k in 1..n and for d | k]).
        """
        return int(generating_series_P(self.size)[-1])

    def get_random_element(self) -> IntegerPartition:
        """
        Generates a uniformly distributed random partition with size n.
        """
        from ..random.random_partitions import UniformPartition

        return UniformPartition(self.size).get_random_element()
