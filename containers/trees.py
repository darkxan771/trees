# Defines the RootedTrees and RecursiveTrees containers
# DoubleRecursiveTrees is a particular instance of RecursiveTrees

from collections.abc import Sequence
from itertools import chain, count

import numpy as np
from scipy.special import factorial

from ..abstraction import InfiniteSetError
from ..boltzmann.boltzmann_tree import generating_series_T
from ..recursive import RecursiveTree
from ..recursive.conversions import permutation_to_code
from ..rooted import RootedTree
from ..rooted.conversions import code_to_nested_list

#############
# Iterators #
#############


class _RecursiveTreesIterator_n:
    def __init__(self, n: int):
        self.order = n
        self.current = np.zeros(n - 1, dtype=int)
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        res = RecursiveTrees().from_code(self.current.tolist())
        test = np.argwhere(self.current < np.arange(self.order - 1))
        if test.size == 0:
            self.finished = True
        else:
            i = test[-1][0]
            self.current[i] += 1
            self.current[i + 1 :] = np.zeros(self.order - i - 2, dtype=int)
        return res


class _DoubleRecursiveTreesIterator_n:
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
        res = RecursiveTrees().from_KP_insertion_array(self.current)
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


class _RootedTreesIterator_n:
    def __init__(self, n: int):
        self.order = n
        self.current = np.arange(n)
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        res = RootedTrees().from_code(self.current.tolist())
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


class RecursiveTrees:
    """
    A container for recursive trees or double recursive trees.

    If a size n is given, the container is restricted to recursive
    trees with size n. In any case, one can iterate upon the container.
    """

    def __init__(self, n: None | int = None, double: bool = False):
        self.order = n
        self.double = double

    def __repr__(self):
        double = ""
        if self.double:
            double = "Double "
        if self.order is None:
            return f"{double}Recursive trees"
        else:
            return f"{double}Recursive trees with size {self.order}"

    def __contains__(self, T):
        A = isinstance(T, RecursiveTree)
        B, C = True, True
        if self.order is not None:
            B = self.order == T.size[0]
        if self.double:
            C = T.is_double_recursive()
        return A and B and C

    def __iter__(self):
        if self.order is not None:
            if self.double:
                return _DoubleRecursiveTreesIterator_n(self.order)
            else:
                return _RecursiveTreesIterator_n(self.order)
        else:
            return chain.from_iterable(
                RecursiveTrees(n, self.double) for n in count(1)
            ).__iter__()

    def cardinality(self) -> int:
        """
        Returns the cardinality of the set of (double) recursive trees.
        """
        n = self.order
        if n is None:
            raise InfiniteSetError("Infinite set")
        if self.double:
            L = np.arange(1, n)
            return int(np.prod(L * (L + 1)) / 2 ** (n - 1))
        else:
            return factorial(n - 1, True)

    def example(self) -> RecursiveTree:
        """
        A recursive tree with 15 vertices.
        """
        if self.order is None:
            code = (0, 0, 0, 0, 1, 1, 0, 4, 2, 1, 6, 1, 3, 4)
            T = RecursiveTree()
            for i in code:
                _ = T.add_node(i)
            return T
        else:
            raise NotImplementedError

    def from_permutation(self, p: Sequence[int]) -> RecursiveTree:
        """
        Constructs the unique recursive tree corresponding to
        the permutation p.
        """
        if self.order is not None and (not len(p) == self.order - 1):
            raise ValueError(
                "p does not have the correct size for the container."
            )
        v = np.array(p)
        v.sort()
        if not np.all(v == np.arange(len(v))):
            raise ValueError("The parameter p is not a permutation")
        return self.from_code(permutation_to_code(np.array(p)).tolist())

    def from_code(self, c: Sequence[int]) -> RecursiveTree:
        """
        Constructs the unique recursive tree corresponding to
        the code c.
        """
        if self.order is not None and (not len(c) == self.order - 1):
            raise ValueError(
                "c does not have the correct size for the container."
            )
        res = RecursiveTree(max_size=len(c) + 1)
        res.weight[0] = len(c) + 1
        for k in range(len(c)):
            _ = res.add_node(c[k], new_weight=len(c) - k)
        return res

    def from_KP_insertion_array(self, L: np.ndarray) -> RecursiveTree:
        """
        Constructs the unique double recursive tree corresponding to
        the insertion array L.

        Each column of L is a pair (i, J) corresponding to the insertion
        of a new node with weight J above the node with weight i.
        """
        if self.order is not None and (not L.shape == (2, self.order - 1)):
            raise ValueError(
                "The insertion array does not have the right size."
            )
        T = RecursiveTree(max_size=L.shape[1] + 1)
        for v in L.transpose()[:,]:
            _ = T.KP_insertion_at_weight(v[0], v[1])
        return T


class RootedTrees:
    """
    A container for rooted (unlabelled) trees.

    If a size n is given, the container is restricted to rooted
    trees with size n. In any case, one can iterate upon the container.
    """

    def __init__(self, n: None | int = None):
        self.order = n

    def __repr__(self):
        if self.order is None:
            return str("Rooted trees")
        else:
            return f"Rooted trees with size {self.order}"

    def __contains__(self, T):
        A = isinstance(T, RootedTree)
        if self.order is None:
            return A
        else:
            return A and (T.size == self.order)

    def __iter__(self):
        if self.order is not None:
            return _RootedTreesIterator_n(self.order)
        else:
            return chain.from_iterable(
                RootedTrees(n) for n in count(1)
            ).__iter__()

    def cardinality(self) -> int:
        """
        Returns the cardinality of the set of rooted trees with
        size n (the program raises an error if the set of all rooted
        trees is considered).

        The cardinality satisfies the recurrence relation:

        T[n+1]
        = 1/n sum([d * T[d] * T[n-k+1] for k in 1..n and for d | k]).
        """

        if self.order is None:
            raise InfiniteSetError("Infinite set")
        else:
            return int(generating_series_T(self.order)[-1])

    def example(self) -> RootedTree:
        """
        A rooted tree with 15 vertices.
        """
        if self.order is None:
            nested = [[[[[[[[]], []], [], []], [[]], []], []]]]
            return self.from_nested_list(nested)
        else:
            raise NotImplementedError

    def from_code(self, L: Sequence[int]) -> RootedTree:
        """
        Returns the unique rooted tree with given level sequence.
        """
        return self.from_nested_list(code_to_nested_list(L))

    def from_nested_list(self, L: list) -> RootedTree:
        """
        Returns the rooted tree corresponding to the nested list.
        """
        return RootedTree([RootedTrees().from_nested_list(k) for k in L])
