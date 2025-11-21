from ..abstraction.partition import IntegerPartition


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
