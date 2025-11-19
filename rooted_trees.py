import numpy as np

from scipy.special import factorial
from collections import defaultdict

from .conversions import nested_list_to_code
from .recursive_trees import RecursiveTree
from .partitions import IntegerPartition


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
        return nested_list_to_code(self.to_nested_list())

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

    ##############
    # Properties #
    ##############
    @property
    def number_of_edges(self) -> int:
        """
        Yhe number of edges of the tree.
        """
        return self.size - 1

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
        code = np.array(self.to_code())
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
    def d(self) -> int:
        """
        The number of increasing labellings of the tree.
        """
        T = self.to_recursive_tree()
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

    #################
    # Visualisation #
    #################

    def plot(self, **options) -> None:
        """
        Plots the rooted tree.
        """
        opt = options
        opt["labels"] = "empty"
        T = self.to_recursive_tree()
        T.plot(**opt)
