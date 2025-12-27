# Defines an abstract class RandomTree, and various subclasses

# TODO: general fragmentation trees (we might need to introduce exchangeable set
# partitions).

# TODO: understand where the longest path falls.

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from ..abstraction.random import Random
from ..boltzmann.boltzmann_tree import compute_values
from ..boltzmann.boltzmann_tree import find_x_for_n
from ..boltzmann.boltzmann_tree import sampler_with_precomputed
from ..containers import RecursiveTrees
from ..containers import RootedTrees
from ..recursive import RecursiveTree
from ..rooted import RootedTree


class UniformRootedTree(Random):
    """
    Class of uniformly distributed random rooted trees with n nodes.
    """

    def __init__(self, n: int):
        self.size = n
        self.object = "rooted tree"
        self.name = "uniform"
        self.parameter = None

    @property
    def label(self):
        return self.size

    def container(self):
        return RootedTrees(self.size)

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


class RandomRecursiveTree(Random):

    @property
    def label(self):
        return self.size

    def container(self):
        return RecursiveTrees(self.size)


class DeterministicRecursiveTree(RandomRecursiveTree):
    """
    A deterministic recursive tree.
    """

    def __init__(self, T: RecursiveTree):
        self.size = T.size
        self.object = "recursive tree"
        self.name = "deterministic"
        self.parameter = T

    def __repr__(self):
        return repr(self.parameter)

    @property
    def label(self):
        return self.parameter


class RandomSubtree(RandomRecursiveTree):
    """
    Random subtree T of a supertree U, which can itself be random (but with
    fixed size).
    """

    def __init__(self, U: RandomRecursiveTree):
        self.size = None
        self.object = "recursive tree"
        self.name = "subtree"
        self.parameter = U

    def __repr__(self):
        return f"Random subtree of a {self.parameter}"

    @property
    def label(self):
        return self.parameter

    def distribution(self) -> defaultdict:
        d = defaultdict(float)
        n = self.parameter.size
        for k in range(n):
            for T in RecursiveTrees(n):
                d[T.subtree(k).convert("code")] += (
                    self.parameter.probability(T) / n
                )
        return d


class RandomCut(RandomRecursiveTree):
    """
    Random cut T of a supertree U, which can itself be random (but with
    fixed size).
    """

    def __init__(self, U: RandomRecursiveTree):
        self.size = None
        self.object = "recursive tree"
        self.name = "cut"
        self.parameter = U

    def __repr__(self):
        return f"Random cut of a {self.parameter}"

    @property
    def label(self):
        return self.parameter

    def distribution(self) -> defaultdict:
        d = defaultdict(float)
        n = self.parameter.size
        for k in range(1, n):
            for T in RecursiveTrees(n):
                code = T.cut(T.subtree_indices(k)).convert("code")
                d[code] += self.parameter.probability(T) / (n - 1)
        return d


class PlancherelRecursiveTree(RandomRecursiveTree):
    """
    Class of Plancherel-distributed random recursive trees with n nodes.
    """

    def __init__(self, n: int):
        self.size = n
        self.object = "recursive tree"
        self.name = "plancherel"
        self.parameter = None


class UniformRecursiveTree(RandomRecursiveTree):
    """
    Class of uniformly distributed random recursive trees with n nodes.
    """

    def __init__(self, n: int):
        self.size = n
        self.object = "recursive tree"
        self.name = "uniform"
        self.parameter = None


class WeightedRecursiveTree(RandomRecursiveTree):
    """
    Class of random recursive trees with n nodes, chosen according to
    weights given by a function i -> w(i).
    """

    def __init__(self, n: int, weight: Callable[[int], float]):
        self.size = n
        self.object = "recursive tree"
        self.name = "weighted"
        self.parameter = weight


class EwensRecursiveTree(RandomRecursiveTree):
    """
    Class of random recursive trees with n nodes, where the partition
    of [1, n-1] associated to subtrees is recursively chosen according
    to the Ewens measure on set partitions.
    """

    def __init__(self, n: int, theta: float = 1):
        self.size = n
        self.object = "recursive tree"
        self.name = "ewens"
        self.parameter = theta
