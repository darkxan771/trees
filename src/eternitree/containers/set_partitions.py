# Defines a SetPartitions container

import numpy as np

from ..abstraction.set_partition import SetPartition


def bell_number(n):
    k = 1
    row = np.array([1])
    while k < n:
        new_row = np.zeros(k + 1, dtype=int)
        new_row[0] = row[-1]
        for i in range(1, k + 1):
            new_row[i] = row[i - 1] + new_row[i - 1]
        row = new_row
        k += 1
    return int(row[-1])


class _SetPartitionsIterator_n:
    def __init__(self, L: list[int]):
        self.set = L
        self.set.sort()
        self.order = len(L)
        self.current = np.zeros((2, self.order), dtype=int)
        self.current[1, 0] = -1
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        n = self.order
        k = np.max(self.current)
        res = [
            [self.set[x] for x in range(self.order) if self.current[0, x] == t]
            for t in range(k + 1)
        ]
        if np.all(
            self.current
            == np.array([np.arange(self.order), np.arange(-1, self.order - 1)])
        ):
            self.finished = True
        else:
            for i in range(n - 1, -1, -1):
                if self.current[0, i] <= self.current[1, i]:
                    self.current[0, i] += 1
                    self.current[0, i + 1 :] = np.zeros(n - i - 1, dtype=int)
                    self.current[1, i + 1 :] = max(
                        self.current[0, i], self.current[1, i]
                    ) * np.ones(n - i - 1, dtype=int)
                    break
        return SetPartition(res)


class SetPartitions:
    """
    A container for set partitions of a given finite set of integers.
    """

    def __init__(self, L: list[int]):
        self.set = L
        self.set.sort()

    def __repr__(self):
        return f"Set partitions of {self.set}"

    def __contains__(self, SP: SetPartition):
        return isinstance(SP, SetPartition) and SP.set == self.set

    def __iter__(self):
        return _SetPartitionsIterator_n(self.set)

    @property
    def order(self):
        """
        The size of the set underlying the set partitions.
        """
        return len(self.set)

    @property
    def cardinality(self) -> int:
        """
        The cardinality of the set of set partitions.
        """
        return bell_number(self.order)

    def example(self) -> SetPartition:
        """
        An example of set partition.
        """
        from ..random.random_partitions import UniformSetPartition

        return UniformSetPartition(self.set).get_random_element()
