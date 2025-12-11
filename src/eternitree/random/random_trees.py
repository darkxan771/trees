# Defines an abstract class RandomTree, and various subclasses

# TODO: general fragmentation trees (we might need to introduce exchangeable set
# partitions).

# TODO: understand where the longest path falls.

from __future__ import annotations

from collections import defaultdict
from typing import Any
from typing import Callable
from typing import Protocol

from ..abstraction import InfiniteSetError
from ..abstraction import Tree
from ..boltzmann.boltzmann_tree import compute_values
from ..boltzmann.boltzmann_tree import find_x_for_n
from ..boltzmann.boltzmann_tree import sampler_with_precomputed
from ..containers import RecursiveTrees
from ..containers import RootedTrees
from ..recursive import RecursiveTree
from ..rooted import RootedTree
from .tree_generators import compute_random_trees
from .tree_probabilities import compute_probabilities


class RandomTree(Protocol):
    """
    A generic class for distributions of random trees (either rooted
    unlabelled trees, or rooted recursive trees).
    """

    size: int | None
    treetype: str = "recursive"
    name: str = "Random"
    parameter: Any = None

    def __repr__(self):
        res = f"{self.name} {self.treetype} tree"
        if self.size is not None:
            res += f" with size {self.size}"
        return res

    def container(self) -> RootedTrees | RecursiveTrees:
        """
        The support of the distribution of random trees.
        """
        if self.treetype == "recursive":
            return RecursiveTrees(self.size)
        else:
            return RootedTrees(self.size)

    def probability(self, T: Tree) -> float:
        """
        Computes the probability of T under the prescribed distribution.
        """
        if T in self.container():
            try:
                return compute_probabilities[self.name, self.treetype](
                    T, self.parameter
                )
            except KeyError:
                return self.distribution()[T.convert("code")]
        else:
            return float(0)

    def distribution(self) -> defaultdict:
        """
        Returns a dictionary with items (T, probability[T]), where T is
        identified by its code.
        """
        if self.size is None:
            raise NotImplementedError
        return defaultdict(
            float,
            {T.convert("code"): self.probability(T) for T in self.container()},
        )

    def distribution_partition(self) -> defaultdict:
        """
        Returns a dictionary with items (L, probability[L]), where L
        runs over the set of integer partitions with size n - 1, and
        probability[L] is the probability of L being the list of
        sizes of the subtrees of a random tree.
        """
        if self.size is None:
            raise InfiniteSetError
        res = defaultdict(float)
        for T in self.container():
            p = tuple(T.subtrees_partition.parts)
            res[p] += self.probability(T)
        return res

    def get_random_element(self) -> RootedTree | RecursiveTree:
        """
        Picks a tree at random under the prescribed distribution.
        """
        return compute_random_trees[self.name, self.treetype](
            self.size, self.parameter
        )


class DeterministicRecursiveTree(RandomTree):
    """
    A deterministic recursive tree.
    """

    def __init__(self, T: RecursiveTree):
        self.size = T.size[0]
        self.treetype = "recursive"
        self.name = "Deterministic"
        self.parameter = T

    def __repr__(self):
        return self.parameter.__repr__()


class RandomSubtree(RandomTree):
    """
    Random subtree T of a supertree U, which can itself be random (but with
    fixed size).
    """

    def __init__(self, U: RandomTree):
        self.size = 0
        self.parameter = U
        if not U.treetype == "recursive":
            raise ValueError("U is not a random tree with recursive treetype")

    def __repr__(self):
        return f"Random subtree of a {self.parameter}"

    def distribution(self) -> defaultdict:
        """
        Returns a dictionary with items (T, probability[T]), where T is
        identified by its code.
        """
        d = defaultdict(float)
        n = self.parameter.size
        for k in range(n):
            for T in RecursiveTrees(n):
                d[T.subtree(k).convert("code")] += (
                    self.parameter.probability(T) / n
                )
        return d


class RandomCut(RandomTree):
    """
    Random cut T of a supertree U, which can itself be random (but with
    fixed size).
    """

    def __init__(self, U: RandomTree):
        self.size = 0
        self.treetype = "recursive"
        self.name = "Cut"
        self.parameter = U
        if not U.treetype == "recursive":
            raise ValueError("T is not a random tree with recursive treetype")

    def __repr__(self):
        return f"Random cut of a {self.parameter}"

    def distribution(self) -> defaultdict:
        """
        Returns a dictionary with items (T, probability[T]), where T is
        identified by its code.
        """
        d = defaultdict(float)
        n = self.parameter.size
        for k in range(1, n):
            for T in RecursiveTrees(n):
                code = T.cut(k).convert("code")
                d[code] += self.parameter.probability(T) / (n - 1)
        return d


class UniformRootedTree(RandomTree):
    """
    Class of uniformly distributed random rooted trees with n nodes.
    """

    def __init__(self, n: int):
        self.size = n
        self.treetype = "rooted"
        self.name = "Uniform"

    def get_random_element(self, exact: bool = True) -> RootedTree:
        """
        Picks at random a rooted unlabelled tree with size n.

        If the parameter exact is set to False, the size n is replaced by a
        random size N in [0.9n, 1.1n]. Conditionnally to N, the distribution
        of the tree is uniform over the set of rooted unlabelled trees with
        size N.
        """
        n = self.size
        if n is None:
            raise NotImplementedError
        elif n == 1:
            return RootedTree([])
        elif n == 2:
            return RootedTree([RootedTree([])])
        else:

            def size_of_nested_list(L: list) -> int:
                return 1 + sum([size_of_nested_list(x) for x in L])

            def tree_of_nested_list(L: list) -> RootedTree:
                return RootedTree([tree_of_nested_list(x) for x in L])

            x = find_x_for_n(n, True)
            values = compute_values(x)
            test = True
            res = []
            while test:
                res = sampler_with_precomputed(values, 1, True)
                S = size_of_nested_list(res)
                if ((not exact) and 0.9 < S / n < 1.1) or (S == n):
                    test = False
            return tree_of_nested_list(res)


class PlancherelRecursiveTree(RandomTree):
    """
    Class of Plancherel-distributed random recursive trees with n nodes.
    """

    def __init__(self, n: int):
        self.size = n
        self.treetype = "recursive"
        self.name = "Plancherel"


class UniformRecursiveTree(RandomTree):
    """
    Class of uniformly distributed random recursive trees with n nodes.
    """

    def __init__(self, n: int):
        self.size = n
        self.treetype = "recursive"
        self.name = "Uniform"

        self.name = "Uniform"


class WeightedRecursiveTree(RandomTree):
    """
    Class of random recursive trees with n nodes, chosen according to
    weights given by a function i -> w(i).
    """

    def __init__(self, n: int, weight: Callable[[int], float]):
        self.size = n
        self.treetype = "recursive"
        self.name = "Weighted"
        self.parameter = weight


class EwensRecursiveTree(RandomTree):
    """
    Class of random recursive trees with n nodes, where the partition
    of [1, n-1] associated to subtrees is recursively chosen according
    to the Ewens measure on set partitions.
    """

    def __init__(self, n: int, theta: float = 1):
        self.size = n
        self.treetype = "recursive"
        self.name = "Ewens"
        self.parameter = theta

    def __repr__(self):
        A = "Ewens recursive tree with size"
        return f"{A} {self.size} and parameter {self.parameter}"
