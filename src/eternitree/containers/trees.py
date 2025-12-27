# Defines the RootedTrees, RecursiveTrees and
# DoubleRecursiveTrees containers

from typing import Callable
from typing import Iterator

import numpy as np

from ..abstraction import CombinatorialClass
from ..recursive import RecursiveTree
from ..rooted import RootedTree
from .generating_series import generating_series_DRT
from .generating_series import generating_series_RT
from .generating_series import generating_series_T

#############
# Iterators #
#############


class _RecursiveTreesIterator_n(Iterator):
    def __init__(self, n: int):
        self.order = n
        self.current = np.zeros(n - 1, dtype=int)
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        res = RecursiveTree.from_code(self.current.tolist())
        test = np.argwhere(self.current < np.arange(self.order - 1))
        if test.size == 0:
            self.finished = True
        else:
            i = test[-1][0]
            self.current[i] += 1
            self.current[i + 1 :] = np.zeros(self.order - i - 2, dtype=int)
        return res


class _DoubleRecursiveTreesIterator_n(Iterator):
    def __init__(self, n: int):
        self.order = n
        self.current = np.ones((2, n - 1), dtype=int)
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        n = self.order
        res = RecursiveTree.from_KP_insertion_array(self.current)
        test_i = np.argwhere(self.current[0, :] < np.arange(1, n))
        test_j = np.argwhere(self.current[1, :] < self.current[0, :])
        if test_j.size == 0:
            if test_i.size == 0:
                self.finished = True
            else:
                i = test_i[-1][0]
                self.current[0, i] += 1
                self.current[0, i + 1 :] = np.ones(n - i - 2, dtype=int)
                self.current[1, :] = np.ones(n - 1, dtype=int)
        else:
            j = test_j[-1][0]
            self.current[1, j] += 1
            self.current[1, j + 1 :] = np.ones(n - j - 2, dtype=int)
        return res


class _RootedTreesIterator_n(Iterator):
    def __init__(self, n: int):
        self.order = n
        self.current = np.arange(n)
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        res = RootedTree.from_code(self.current.tolist())
        if sum(self.current) == self.order - 1:
            self.finished = True
        else:
            p = np.argwhere(self.current > 1)[-1][0]
            q = np.argwhere(self.current[:p] == self.current[p] - 1)[-1][0]
            for i in range(p, self.order):
                self.current[i] = self.current[i - (p - q)]
        return res


##############
# Containers #
##############


class RecursiveTrees(CombinatorialClass):
    """
    A container for recursive trees.

    If a size n is given, the container is restricted to recursive
    trees with size n. In any case, one can iterate upon the container.
    """

    def __init__(self, n: int | None = None):
        self.category = "recursive tree"
        self.order = n

    @classmethod
    def iter_n(cls) -> Callable[[int], Iterator]:
        return lambda n: _RecursiveTreesIterator_n(n)

    @classmethod
    def generating_series(cls, N: int) -> list[int]:
        return generating_series_RT(N)

    @staticmethod
    def example() -> RecursiveTree:
        """
        An example of recursive tree.
        """
        code = (0, 0, 0, 0, 1, 1, 0, 4, 2, 1, 6, 1, 3, 4)
        T = RecursiveTree()
        for i in code:
            _ = T.add_node(i)
        return T


class DoubleRecursiveTrees(CombinatorialClass):
    """
    A container for double recursive trees.

    If a size n is given, the container is restricted to double recursive
    trees with size n. In any case, one can iterate upon the container.
    """

    def __init__(self, n: int | None = None):
        self.category = "recursive tree"
        self.order = n

    def __repr__(self) -> str:
        return "Double " + super().__repr__().lower()

    def __contains__(self, obj) -> bool:
        A = super().__contains__(obj)
        return A and obj.is_double_recursive()

    @classmethod
    def iter_n(cls) -> Callable[[int], Iterator]:
        return lambda n: _DoubleRecursiveTreesIterator_n(n)

    @classmethod
    def generating_series(cls, N: int) -> list[int]:
        return generating_series_DRT(N)

    @staticmethod
    def example() -> RecursiveTree:
        """
        An example of double recursive tree.
        """
        code = (0, 0, 0, 0, 1, 1, 0, 4, 2, 1, 6, 1, 3, 4)
        T = RecursiveTree()
        T.weight[0] = 15
        for k, i in enumerate(code):
            _ = T.add_node(i, new_weight=14 - k)
        return T


class RootedTrees(CombinatorialClass):
    """
    A container for rooted (unlabelled) trees.

    If a size n is given, the container is restricted to rooted
    trees with size n. In any case, one can iterate upon the container.
    """

    def __init__(self, n: None | int = None):
        self.category = "rooted tree"
        self.order = n

    @classmethod
    def iter_n(cls) -> Callable[[int], Iterator]:
        return lambda n: _RootedTreesIterator_n(n)

    @classmethod
    def generating_series(cls, N: int) -> list[int]:
        return generating_series_T(N)

    @staticmethod
    def example() -> RootedTree:
        """
        An example of rooted tree.
        """
        nested = [[[[[[[[]], []], [], []], [[]], []], []]]]
        return RootedTree.from_nested_list(nested)
