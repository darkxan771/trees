# Defines the abstract SetPartition class

# TODO: tests for set partitions / documentation of methods.


from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Iterable
from typing import Self

from .helpers import shift_dict
from .partition import IntegerPartition

compute_conversions: dict[str, Callable] = {
    "code": lambda P: tuple(tuple(p) for p in P.parts),
    "dict": lambda P: P.dict,
}


class SetPartition:
    """
    A class for the manipulation of set partitions of integer sets.

    Internally, a set partition P is saved as a dictionary {k: P[k]},
    where k runs over the set of indices of the parts, and P[k]
    is the k-th part, saved as a sorted list of integers. The parts
    are themselves sorted according to their minimal element.
    """

    def __init__(self, L: list[list[int]]):
        self.dict = {}
        S = [x for p in L for x in p]
        S.sort()
        if any([len(p) == 0 for p in L]):
            raise ValueError("One of the parts is empty.")
        if any([S[i] == S[i + 1] for i in range(len(S) - 1)]):
            raise ValueError("The parts are not disjoint.")
        M = L.copy()
        M.sort(key=lambda p: min(p))
        for k in range(len(L)):
            self.dict[k] = M[k]
            self.dict[k].sort()

    def __repr__(self):
        return f"Set partition {self.dict}"

    def __call__(self, k: int):
        return self.dict[k]

    def __eq__(self, other):
        A = isinstance(other, SetPartition)
        B = self.length == other.length
        return A and B and all(self(k) == other(k) for k in range(self.length))

    @property
    def convert(self) -> Callable[[str], Any]:
        return lambda typ: compute_conversions[typ](self)

    @property
    def length(self) -> int:
        """
        The length of the set partition (number of parts).
        """
        return len(list(self.dict.keys()))

    @property
    def set(self) -> list:
        """
        The set underlying the set partition.
        """
        res = sum([self(k) for k in range(self.length)], [])
        res.sort()
        return res

    @property
    def size(self) -> int:
        """
        The size of the set partition (sum of the sizes of its parts).
        """
        return len(self.set)

    @property
    def composition(self) -> list[int]:
        """
        The lengths of the parts of the set partition; the parts are
        ordered according to their minima.
        """
        return [len(self(k)) for k in range(self.length)]

    @property
    def type(self) -> IntegerPartition:
        """
        The lengths of the parts of the set partition, reordered in
        a non-increasing sequence (integer partition.)
        """
        c = self.composition
        c.sort(reverse=True)
        return IntegerPartition(c)

    @property
    def parts(self) -> Iterable:
        """
        The parts of the set partition.
        """
        return (self(k) for k in range(self.length))

    def find(self, x: int) -> int:
        """
        Finds the index of the part of the set partition which contains x.
        """
        if not x in self.set:
            raise ValueError(f"{x} is not in the set partition")
        else:
            res = [x in p for p in self.parts]
            return res.index(True)

    def remove(self, x: int) -> Self:
        """
        Removes x from the corresponding part of the set partition.
        Said part is deleted if it becomes empty.
        """
        k = self.find(x)
        self.dict[k].remove(x)
        if self.dict[k] == []:
            n = self.length
            self.dict = shift_dict(self.dict, k + 1, n, -1)
        return self

    def add_part(self, x: int) -> Self:
        """
        Adds a new part [x] to the set partition.
        """
        if x in self.set:
            raise ValueError(f"{x} is already in the set partition")
        minima = [p[0] for p in self.parts]
        k = len([y for y in minima if y < x])
        n = self.length
        res = {}
        for j in range(k):
            res[j] = self(j)
        res[k] = [x]
        for j in range(k, n):
            res[j + 1] = self(j)
        self.dict = res
        return self

    def restrict(self, I: Iterable[int]) -> SetPartition:
        """
        Restricts all the parts of the set partition to their
        intersection with I.
        """
        L = [x for x in I]
        L.sort()
        if any(x not in self.set for x in L):
            raise ValueError("I is not a subset of the set of the partition")
        res = [[x for x in p if x in I] for p in self.parts]
        return SetPartition(res)
