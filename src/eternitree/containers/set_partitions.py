# Defines a SetPartitions container

from reprlib import Repr
from typing import Callable
from typing import Iterator
from typing import Sequence

import numpy as np

from ..abstraction import CombinatorialClass
from ..abstraction import SetPartition
from .generating_series import generating_series_SP


class _SetPartitionsIterator_n(Iterator):
    def __init__(self, L: list[int]):
        self.set = L.copy()
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


class SetPartitions(CombinatorialClass):
    """
    A container for set partitions of a given finite set of integers.
    """

    def __init__(self, L: int | Sequence[int] | None = None):
        if L is None:
            self.order = None
            self.is_standard = True
        elif isinstance(L, int):
            self.set = list(range(1, L + 1))
            self.order = L
            self.is_standard = True
        else:
            self.set = list(L)
            self.set.sort()
            self.order = len(self.set)
            self.is_standard = self.set == list(range(1, self.order + 1))
        self.category = "set partition"

    @classmethod
    def generating_series(cls, N: int) -> list[int]:
        return generating_series_SP(N)

    @classmethod
    def iter_n(cls) -> Callable[[int], Iterator]:
        return lambda n: _SetPartitionsIterator_n(list(range(1, n + 1)))

    def __iter__(self) -> Iterator:
        if self.is_standard:
            return super().__iter__()
        else:
            return _SetPartitionsIterator_n(self.set)

    def __repr__(self) -> str:
        if self.is_standard:
            return super().__repr__()
        else:
            return f"Set partitions of {Repr(maxlist=10).repr(self.set)}"

    def __contains__(self, SP) -> bool:
        if self.is_standard:
            return SP.is_standard and super().__contains__(SP)
        else:
            return SP.set == self.set

    @staticmethod
    def example() -> SetPartition:
        """
        An example of set partition.
        """
        return SetPartition([[1, 3, 5, 6], [2, 9], [4, 8, 11], [10], [7, 12]])
