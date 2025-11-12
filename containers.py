import numpy as np

from .recursive_trees import RecursiveTree
from .rooted_trees import RootedTree
from .random_trees import (
    UniformRootedTree,
    UniformRecursiveTree,
    PlancherelRecursiveTree,
)
from .routines import permutation_to_code, code_to_nested_list
from scipy.special import factorial


class _RecursiveTreesIterator:
    def __init__(self, n: int):
        self.order = n
        self.current = np.zeros(n - 1, dtype=int)
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        res = RecursiveTrees_all().from_code(self.current)
        test = np.argwhere(self.current < np.arange(self.order - 1))
        if test.size == 0:
            self.finished = True
        else:
            i = test[-1][0]
            self.current[i] += 1
            self.current[i + 1 :] = np.zeros(self.order - i - 2, dtype=int)
        return res


class _DoubleRecursiveTreesIterator:
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
        res = RecursiveTrees_all().from_KP_insertion_array(self.current)
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


class RecursiveTrees_all:
    """
    A container for all recursive trees.
    """

    def __init__(self):
        self.order = None

    def __repr__(self):
        return str("Recursive trees")

    def __contains__(self, T):
        return isinstance(T, RecursiveTree)

    def from_permutation(self, p: tuple | list | np.ndarray) -> RecursiveTree:
        """
        Constructs the unique recursive tree corresponding to
        the permutation p.
        """
        return self.from_code(permutation_to_code(np.array(p)))

    def from_code(self, c: tuple | list | np.ndarray) -> RecursiveTree:
        """
        Constructs the unique recursive tree corresponding to
        the code c.
        """
        res = RecursiveTree(max_size=len(c) + 1)
        for i in c:
            res.add_node(i)
        return res

    def from_KP_insertion_array(self, L: np.ndarray) -> RecursiveTree:
        """
        Constructs the unique double recursive tree corresponding to
        the insertion array L.

        Each column of L is a pair (i, J) corresponding to the insertion
        of a new node with weight J above the node with weight i.
        """
        T = RecursiveTree(max_size=L.shape[1] + 1)
        for v in L.transpose()[:,]:
            T.KP_insertion_at_weight(v[0], v[1])
        return T


class RecursiveTrees_n(RecursiveTrees_all):
    """
    A container for recursive or double recursive trees with given size n.
    """

    def __init__(self, n: int, double: bool = False):
        self.order = n
        self.double = double

    def __repr__(self):
        if self.double:
            return f"Double recursive trees with size {self.order}"
        else:
            return f"Recursive trees with size {self.order}"

    def __iter__(self):
        if self.double:
            return _DoubleRecursiveTreesIterator(self.order)
        else:
            return _RecursiveTreesIterator(self.order)

    def __contains__(self, T):
        A = isinstance(T, RecursiveTree) and (T.size[0] == self.order)
        if self.double:
            return A and T.is_double_recursive()
        else:
            return A

    def cardinality(self) -> int:
        """
        Returns the cardinality of the set of (double) recursive trees.
        """
        n = self.order
        if self.double:
            L = np.arange(1, n)
            return int(np.prod(L * (L + 1)) / 2 ** (n - 1))
        else:
            return factorial(n - 1, True)

    def get_random_element(
        self, distribution: str = "uniform"
    ) -> RecursiveTree:
        """
        Picks a recursive tree at random. Available distributions are:
        "uniform", "plancherel".
        """
        if distribution == "plancherel":
            return PlancherelRecursiveTree(self.order).get_random_element()
        else:
            return UniformRecursiveTree(self.order).get_random_element()


def RecursiveTrees(
    n: int | None = None, double: bool = False
) -> RecursiveTrees_all | RecursiveTrees_n:
    """
    Returns:
    - the class of all recursive trees if no parameter n is given,
    - the class of all recursive trees with size n if an integer n
    is given, but 'double' is False (default).
    - the class of all double recursive trees with size n if an
    integer n is given and 'double' is True.
    """
    if n is None:
        return RecursiveTrees_all()
    else:
        return RecursiveTrees_n(n, double)


class _RootedTreesIterator:
    def __init__(self, n: int):
        self.order = n
        self.current = np.arange(n)
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        res = RootedTrees_all().from_code(self.current)
        if sum(self.current) == self.order - 1:
            self.finished = True
        else:
            p = np.argwhere(self.current > 1)[-1][0]
            q = np.argwhere(self.current[:p] == self.current[p] - 1)[-1][0]
            for i in range(p, self.order):
                self.current[i] = self.current[i - (p - q)]
        return res


class RootedTrees_all:
    """
    A container for all rooted (unlabelled) trees.
    """

    def __init__(self):
        self.order = None

    def __repr__(self):
        return str("Rooted trees")

    def __contains__(self, T):
        return isinstance(T, RootedTree)

    def from_code(self, L: tuple | list | np.ndarray) -> RootedTree:
        """
        Returns the unique rooted tree with given level sequence.
        """
        return self.from_nested_list(code_to_nested_list(L))

    def from_nested_list(self, L) -> RootedTree:
        """
        Returns the rooted tree corresponding to the nested list.
        """
        return RootedTree([RootedTrees_all().from_nested_list(k) for k in L])


class RootedTrees_n(RootedTrees_all):
    """
    A container for rooted (unlabelled) trees with a given size n.
    """

    def __init__(self, n: int):
        self.order = n

    def __repr__(self):
        return f"Rooted trees with size {self.order}"

    def __iter__(self):
        return _RootedTreesIterator(self.order)

    def __contains__(self, T):
        return isinstance(T, RootedTree) and T.size == self.order

    def cardinality(self) -> int:
        """
        Returns the cardinality of the set of rooted trees with size n.

        The cardinality satisfies the recurrence relation:

        C[n+1]
        = 1/n sum([d * C[d] * C[n-k+1] for k in 1..n and for d | k]).
        """
        from .boltzmann import generating_series_T

        return int(generating_series_T(self.order)[-1])

    def get_random_element(self) -> RootedTree:
        """
        Generates a uniformly distributed random rooted tree with size n.
        """
        return UniformRootedTree(self.order).get_random_element()


def RootedTrees(n: int | None = None) -> RootedTrees_all | RootedTrees_n:
    """
    Returns:
    - the class of all rooted trees if no parameter is given,
    - the class of all rooted trees with size n if an integer n
    is given.
    """
    if n is None:
        return RootedTrees_all()
    else:
        return RootedTrees_n(n)
