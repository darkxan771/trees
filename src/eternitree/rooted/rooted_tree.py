# Defines the class RootedTree


from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
from scipy.special import factorial

from ..abstraction import Tree


class RootedTree(Tree):
    """
    A rooted tree is a (possibly empty) multiset of rooted
    trees, which are connected to a common root vertex.
    """

    def __init__(self, subtrees: list = []):
        self.subtrees = list(subtrees)
        self.subtrees.sort(key=(lambda x: x.code), reverse=True)
        self.size = 1 + sum([child.size for child in self.subtrees])
        from ..abstraction.conversions import rooted_to_code

        self.code = rooted_to_code(self)

    def __abs__(self):
        return self.size

    def __len__(self):
        if self.subtrees == []:
            return 0
        else:
            return 1 + max([child.height for child in self.subtrees])

    ##############
    # Properties #
    ##############

    @property
    def type(self) -> str:
        return "rooted"

    @property
    def profile(self) -> np.ndarray:
        """
        The profile of the tree (number of nodes on each level).
        """
        code = np.array(self.code)
        h = max(code)
        return np.array([np.count_nonzero(code == d) for d in range(h + 1)])

    @property
    def subtree_list(self) -> list:
        """
        The list of subtrees of the tree.
        """
        return self.subtrees

    @property
    def weights(self) -> np.ndarray:
        return np.ones(self.size, dtype=int)

    @property
    def data(self) -> Callable[[str], np.ndarray]:
        return self.convert("recursive").data

    @property
    def d(self) -> int:
        """
        The number of increasing labellings of the tree.
        """
        T = self.convert("recursive")
        return int(factorial(self.size) / np.prod(T.size[: self.size]))

    @property
    def sym(self) -> int:
        """
        The symmetry factor of the tree.
        """
        if self.size == 0:
            return 1
        else:
            m = defaultdict(int)
            for child in self.subtrees:
                m[child.code] += 1
            prod1 = np.prod(np.array([factorial(m[k]) for k in m]))
            prod2 = np.prod(np.array([child.sym for child in self.subtrees]))
            return int(prod1 * prod2)

    @property
    def u(self) -> int:
        """
        The number of increasing labellings of the tree,
        up to isomorphisms.
        """
        return int(self.d / self.sym)

    @property
    def plancherel_measure(self) -> float:
        """
        The Plancherel measure of the rooted tree.

        Notice that this differs from the probability given by PlancherelRecursiveTree,
        because this class produces recursive trees instead of rooted trees.
        """
        num = self.d * self.u
        denum = np.prod(
            np.array([(i * (i + 1) / 2) for i in range(1, self.size + 1)])
        )
        return float(num / denum)

    def trim(self, epsilon: float) -> RootedTree:
        """
        Removes all the subtrees with relative size smaller than epsilon.
        """
        return RootedTree(
            [
                child.trim(epsilon)
                for child in self.subtrees
                if child.size >= epsilon * (self.size - 1)
            ]
        )
