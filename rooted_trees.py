import numpy as np

from scipy.special import factorial
from collections import defaultdict

from .routines import c_flatten, raise_tuple
from .recursive_trees import RecursiveTree


class RootedTree:
    """
    A rooted tree is a (possibly empty) multiset of rooted
    trees, which are connected to a common root vertex.
    """

    def __init__(self, children: list = []):
        self.children = list(children)
        self.children.sort(key=(lambda x: x.code), reverse=True)
        self.size = 1 + sum([child.size for child in self.children])
        self.code = self.to_code()

    def __repr__(self):
        return f"Rooted tree of size {self.size}"

    def __hash__(self):
        return hash(self.code)

    def __eq__(self, other):
        return isinstance(other, RootedTree) and (self.code == other.code)

    ###############
    # Conversions #
    ###############

    def to_code(self) -> tuple:
        """
        Returns the code of the tree (level sequence in the depth
        first search).

        It is a tuple of integers which uniquely identifies the tree.
        """
        return tuple(
            c_flatten(
                [(0,)] + [raise_tuple(child.code) for child in self.children]
            )
        )

    def to_nested_list(self) -> list:
        """
        Converts the rooted tree to a nested list.
        """
        return [child.to_nested_list() for child in self.children]

    def _insert_in_recursive_tree(
        self, T, d: int, mini: int, maxi: int
    ) -> None:
        sizes = [c.size for c in self.children]
        T.children[mini] = (
            mini + 1 + np.cumsum(np.array([0] + sizes))[:-1]
        ).tolist()
        T.weight[mini] = 1
        T.depth[mini] = d
        T.size[mini] = maxi - mini
        if len(sizes) > 0:
            mini2 = mini + 1
            for c in self.children:
                maxi2 = mini2 + c.size
                T.parent[mini2] = mini
                c._insert_in_recursive_tree(T, d + 1, mini2, maxi2)
                mini2 = maxi2

    def to_recursive_tree(self) -> RecursiveTree:
        """
        Converts the rooted tree to a recursive tree. The increasing
        labelling which is chosen is uniformly distributed over all
        possibilities.
        """

        T = RecursiveTree(max_size=self.size)
        self._insert_in_recursive_tree(T, 0, 0, self.size)
        T.random_relabelling()
        return T

    ###################################
    # Extract statistical information #
    ###################################

    def number_of_vertices(self) -> int:
        """
        Returns the size of the tree (number of vertices).
        """
        return self.size

    def number_of_edges(self) -> int:
        """
        Returns the number of edges of the tree.
        """
        return self.size - 1

    def height(self) -> int:
        """
        Returns the height of the tree (maximal depth of a node).
        """
        if self.children == []:
            return 0
        else:
            return 1 + max([child.height() for child in self.children])

    def d(self) -> int:
        """
        Returns the number of increasing labellings of the tree.
        """
        T = self.to_recursive_tree()
        return int(factorial(self.size) / np.prod(T.size[: self.size]))

    def sym(self) -> int:
        """
        Returns the symmetry factor of the tree.
        """
        if self.size == 0:
            return 1
        else:
            m = defaultdict(int)
            for child in self.children:
                m[child.code] += 1
            prod1 = np.prod(np.array([factorial(m[k]) for k in m]))
            prod2 = np.prod(np.array([child.sym() for child in self.children]))
            return int(prod1 * prod2)

    def u(self) -> int:
        """
        Returns the number of increasing labellings of the tree,
        up to isomorphisms.
        """
        return int(self.d() / self.sym())

    def plancherel_measure(self) -> float:
        """
        Returns the Plancherel measure of the rooted tree.
        """
        num = self.d() * self.u()
        denum = np.prod(
            np.array([(i * (i + 1) / 2) for i in range(1, self.size + 1)])
        )
        return float(num / denum)

    #################
    # Visualisation #
    #################

    def plot(
        self, style: str = "centered", with_circles: bool = False, **kwargs
    ) -> None:
        """
        Plots the rooted tree.
        """
        T = self.to_recursive_tree()
        T.plot(style, labels="empty", with_circles=with_circles, **kwargs)
