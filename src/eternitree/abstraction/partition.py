# Defines the abstract IntegerPartition class

from __future__ import annotations

from reprlib import Repr
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial

from .object import CombinatorialObject


class IntegerPartition(CombinatorialObject):
    """
    A class for the manipulation of integer partitions.
    """

    def __init__(self, L: Sequence[int]):
        if any(x <= 0 for x in L):
            raise ValueError(
                "The elements of L should be non-negative integers."
            )
        if any(x < y for x, y in zip(L[:-1], L[1:])):
            raise ValueError(
                "The elements of L should be in non-increasing order."
            )
        self.parts = list(int(x) for x in L)
        self.parts

    def __repr__(self) -> str:
        return f"Partition {Repr(maxlist=10).repr(self.parts)}"

    def __abs__(self) -> int:
        return sum(self.parts)

    def __len__(self) -> int:
        return len(self.parts)

    @property
    def category(self) -> str:
        return "partition"

    @property
    def dictionary(self) -> dict[int, int]:
        """
        The multiplicities of the integers as parts of the integer partition.
        """
        s = max(self.parts)
        res = {}
        for i in range(1, s + 1):
            m = self.parts.count(i)
            if m > 0:
                res[i] = m
        return res

    @property
    def bell_number(self) -> int:
        """
        The number of set partitions of [1,n] with type given by the
        integer partition.
        """
        res = factorial(self.size)
        for i, m in self.dictionary.items():
            res /= factorial(m, exact=True) * (factorial(i, exact=True) ** m)
        return int(res)

    @property
    def z(self) -> int:
        """
        The inverse proportion of permutations of [1,n] with cycle type
        given by the integer partition.
        """
        res = 1
        for i, m in self.dictionary.items():
            res *= factorial(m, exact=True) * (i**m)
        return int(res)

    @property
    def conjugate(self) -> IntegerPartition:
        """
        The conjugate integer partition.
        """
        res = [0] * self.parts[0]
        for i in range(self.parts[0]):
            res[i] = int(np.count_nonzero(np.array(self.parts) > i))
        return IntegerPartition(res)

    @property
    def dimension(self) -> int:
        """
        The number of standard tableaux with shape given by the
        integer partition.
        """
        C = self.conjugate.parts
        L = self.parts
        res = int(factorial(self.size, exact=True))
        for j in range(len(L)):
            for i in range(L[j]):
                res //= L[j] - i + C[i] - j - 1
        return res

    def plot(self, style="french") -> None:
        """
        Plots the Young diagram of the integer partition.

        Available options:
        - style: "french", "english", "russian".
        """
        from .plot import draw_partition_on_ax

        _, ax = plt.subplots()
        draw_partition_on_ax(self, ax, style)
        plt.show()
