import numpy as np

from scipy.special import factorial
from collections import defaultdict
from .routines import c_flatten, raise_tuple, code_to_nested_list
from .random_trees import UniformRootedTree


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

    def to_code(self):
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

    def to_nested_list(self):
        """
        Converts the rooted tree to a nested list.
        """
        return [child.to_nested_list() for child in self.children]

    def _insert_in_recursive_tree(self, T, d: int, mini: int, maxi: int):
        sizes = [c.size for c in self.children]
        if not maxi - mini == sum(sizes) + 1:
            raise ValueError
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

    def to_recursive_tree(self):
        """
        Converts the rooted tree to a recursive tree. The increasing
        labelling which is chosen is uniformly distributed over all
        possibilities.
        """
        from .recursive_trees import RecursiveTree

        T = RecursiveTree(max_size=self.size)
        self._insert_in_recursive_tree(T, 0, 0, self.size)
        T.random_relabelling()
        return T

    ###################################
    # Extract statistical information #
    ###################################

    def number_of_vertices(self):
        """
        Returns the size of the tree (number of vertices).
        """
        return self.size

    def number_of_edges(self):
        """
        Returns the number of edges of the tree.
        """
        return self.size - 1

    def height(self):
        """
        Returns the height of the tree (maximal depth of a node).
        """
        if self.children == []:
            return 0
        else:
            return 1 + max([child.height() for child in self.children])

    def d(self):
        """
        Returns the number of increasing labellings of the tree.
        """
        T = self.to_recursive_tree()
        return int(factorial(self.size) / np.prod(T.size[: self.size]))

    def sym(self):
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

    def u(self):
        """
        Returns the number of increasing labellings of the tree,
        up to isomorphisms.
        """
        return int(self.d() / self.sym())

    def plancherel_measure(self):
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
    ):
        """
        Plots the rooted tree.
        """
        T = self.to_recursive_tree()
        T.plot(style, labels="empty", with_circles=with_circles, **kwargs)


class _RootedTreesIterator:
    def __init__(self, n):
        self.order = n
        self.current = np.arange(n)
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        res = RootedTrees(self.order).from_code(self.current)
        if sum(self.current) == self.order - 1:
            self.finished = True
        else:
            p = np.argwhere(self.current > 1)[-1][0]
            q = np.argwhere(self.current[:p] == self.current[p] - 1)[-1][0]
            for i in range(p, self.order):
                self.current[i] = self.current[i - (p - q)]
        return res


class RootedTrees:
    """
    A container for rooted (unlabelled) trees with a given size n.
    """

    def __init__(self, n):
        self.order = n

    def __repr__(self):
        return f"Rooted trees with size {self.order}"

    def __iter__(self):
        return _RootedTreesIterator(self.order)

    def __contains__(self, el):
        return isinstance(el, RootedTree) and el.size == self.order

    def cardinality(self):
        """
        Returns the cardinality of the set of rooted trees with size n.

        The cardinality satisfies the recurrence relation:

        C[n+1]
        = 1/n sum([d * C[d] * C[n-k+1] for k in 1..n and for d | k]).
        """
        from .boltzmann import generating_series_T

        return int(generating_series_T(self.order)[-1])

    def from_code(self, L):
        """
        Returns the unique rooted tree with given level sequence.
        """
        return self.from_nested_list(code_to_nested_list(L))

    def _from_nested_list_without_checking(self, L):
        return RootedTree(
            [RootedTrees(0)._from_nested_list_without_checking(k) for k in L]
        )

    def from_nested_list(self, L, check=True):
        """
        Checks if the nested list has the correct size, and
        returns the corresponding rooted tree.
        """
        T = self._from_nested_list_without_checking(L)
        if check and (not (T.size == self.order)):
            raise ValueError
        return T

    def get_random_element(self):
        """
        Generates a uniformly distributed random rooted tree with size n.
        """
        return UniformRootedTree(self.order).get_random_element()
