# Defines the class RootedTree

import numpy as np

from scipy.special import factorial
from collections import defaultdict
from typing import Any, Callable
from ..abstraction import Tree, IntegerPartition


class RootedTree(Tree):
    """
    A rooted tree is a (possibly empty) multiset of rooted
    trees, which are connected to a common root vertex.
    """

    def __init__(self, children: list = []):
        self.children = list(children)
        self.children.sort(key=(lambda x: x.code), reverse=True)
        self.size = 1 + sum([child.size for child in self.children])
        from .conversions import tree_to_code

        self.code = tree_to_code(self)

    def __repr__(self):
        return f"Rooted tree of size {self.size}"

    def __hash__(self):
        return hash(self.code)

    def __eq__(self, other):
        return isinstance(other, RootedTree) and (self.code == other.code)

    ##############
    # Properties #
    ##############

    @property
    def type(self) -> str:
        return "rooted"

    @property
    def number_of_vertices(self) -> int:
        """
        The number of vertices of the tree.
        """
        return self.size

    @property
    def height(self) -> int:
        """
        The height of the tree (maximal depth of a node).
        """
        if self.children == []:
            return 0
        else:
            return 1 + max([child.height for child in self.children])

    @property
    def profile(self) -> np.ndarray:
        """
        The profile of the tree (number of nodes on each level).
        """
        code = np.array(self.code)
        h = max(code)
        return np.array([np.count_nonzero(code == d) for d in range(h + 1)])

    @property
    def subtrees_partition(self) -> IntegerPartition:
        """
        The integer partition with size n-1 corresponding to the
        sizes of the subtrees attached to the root.
        """
        res = [T.size for T in self.children]
        res.sort(reverse=True)
        return IntegerPartition(res)

    @property
    def weights(self) -> np.ndarray:
        return np.ones(self.size, dtype=int)

    @property
    def convert(self) -> Callable[[str], Any]:
        """
        Converts the rooted tree to another type. Available
        formats are:
        "code", "nested_list", "networkx", "rooted",
        "recursive", "dataframe", "KP_insertion_array".
        """
        from .conversions import compute_conversions

        return lambda str: compute_conversions[str](self)

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
            for child in self.children:
                m[child.code] += 1
            prod1 = np.prod(np.array([factorial(m[k]) for k in m]))
            prod2 = np.prod(np.array([child.sym for child in self.children]))
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
        """
        num = self.d * self.u
        denum = np.prod(
            np.array([(i * (i + 1) / 2) for i in range(1, self.size + 1)])
        )
        return float(num / denum)
