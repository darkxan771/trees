# Defines the class RecursiveTree

from __future__ import annotations
import numpy as np
import numpy.random as rand
from typing import Any, Callable, Self
from ..abstraction import Tree, IntegerPartition
from ..abstraction.helpers import standardisation

compute_data: dict[str, Callable] = {
    "degree": (
        lambda T: np.vectorize(lambda L: len(L))(T.children[: T.size[0]])
    ),
    "depth": (lambda T: T.depth[: T.size[0]]),
    "size": (lambda T: T.size[: T.size[0]]),
}


class RecursiveTree(Tree):
    """
    A class for the encoding of a weighted recursive tree.

    A recursive tree with n nodes is encoded as an array with several
    length n rows.

    - each column i corresponds to the node i in [0, n-1]. The root
      has label i=0.
    - T.parent[i] is the label of the parent of the node i (with,
      by convention, T.parent[0] = -1).
    - T.weight[i] contains a positive integer, the weight of the node i.
      This is used in particular when grafting new nodes randomly to T.
    - T.size[i] is the size of the subtree based at i. In particular,
      T.size[0] = n.
    - T.depth[i] is the depth of the node i (distance to the root). In
      particular, T.depth[0] = 0.
    - T.children[i] contains the list of the children of the node i.

    Additional information can be stocked in T.additional.
    """

    def __init__(self, max_size: int = 10**6):
        self.parent = -np.ones(max_size, dtype=int)
        self.weight = np.zeros(max_size, dtype=int)
        self.weight[0] = 1
        self.size = np.zeros(max_size, dtype=int)
        self.size[0] = 1
        self.depth = -np.ones(max_size, dtype=int)
        self.depth[0] = 0
        self.children = np.empty(max_size, dtype=object)
        self.children[0] = []
        self.limit = max_size
        self.additional = {}

    def __repr__(self):
        return "Recursive tree with size " + str(self.size[0])

    def __hash__(self):
        return hash(self.convert("code"))

    def __eq__(self, other):
        A = isinstance(other, RecursiveTree)
        B = self.convert("code") == other.convert("code")
        return A and B

    ##############
    # Properties #
    ##############

    @property
    def type(self) -> str:
        return "recursive"

    @property
    def number_of_vertices(self) -> int:
        """
        The size of the tree (number of vertices).
        """
        return int(self.size[0])

    @property
    def height(self) -> int:
        """
        The height of the tree (maximal depth of a node).
        """
        return int(np.max(self.depth))

    @property
    def profile(self) -> np.ndarray:
        """
        The profile of the tree (number of nodes on each level).
        """
        h = self.height
        return np.array(
            [np.count_nonzero(self.depth == d) for d in range(h + 1)]
        )

    @property
    def subtrees_partition(self) -> IntegerPartition:
        """
        The integer partition with size n-1 corresponding to the
        sizes of the subtrees attached to the root.
        """
        children = np.array(self.children[0])
        res = self.size[children].tolist()
        res.sort(reverse=True)
        return IntegerPartition(res)

    @property
    def weights(self) -> np.ndarray:
        """
        The weights of the nodes of the recursive tree.
        """
        return self.weight[: self.size[0]]

    @property
    def convert(self) -> Callable[[str], Any]:
        """
        Converts the recursive tree to another type. Available
        formats are:
        "code", "permutation", "networkx", "rooted",
        "recursive", "dataframe", "KP_insertion_array".
        """
        from .conversions import compute_conversions

        return lambda str: compute_conversions[str](self)

    @property
    def data(self) -> Callable[[str], np.ndarray]:
        return lambda name: compute_data[name](self)

    def subtree_indices(self, k: int) -> list[int]:
        """
        Returns the set of indices in the subtree based at k.
        """
        L = sum([self.subtree_indices(int(n)) for n in self.children[k]], [k])
        L.sort()
        return L

    @property
    def row_positions(self) -> np.ndarray:
        """
        The row positions of the nodes.
        """
        res = np.zeros(self.size[0], dtype=int)
        h = self.height
        count = np.zeros(h, dtype=int)
        level = [0]
        next_level = self.children[0]
        for d in range(h):
            for x in next_level:
                res[x] = count[d]
                count[d] += 1
            level = next_level
            next_level = sum([self.children[x] for x in level], [])
        return res

    def path_to_root(self, k: int) -> np.ndarray:
        """
        Returns the unique path from the root to k.
        """
        res = np.zeros(self.depth[k] + 1, dtype=int)
        res[-1] = k
        for i in range(self.depth[k]):
            res[-i - 2] = self.parent[res[-i - 1]]
        return res

    def is_double_recursive(self) -> bool:
        """
        Checks whether the tree is double recursive (i.e.,
        its set of weights is [1, n] and the weights are
        decreasing.
        """
        n = self.size[0]
        if np.all(np.sort(self.weight[:n]) == np.arange(1, n + 1)):
            return bool(
                np.all(self.weight[self.parent[1:n]] > self.weight[1:n])
            )
        else:
            return False

    ###################
    # Transformations #
    ###################

    def map_weights(self, func) -> Self:
        """
        Maps a function on the set of weights.
        """
        n = self.size[0]
        self.weight[:n] = np.vectorize(func)(self.weight[:n])
        return self

    def resize(self, k) -> Self:
        """
        Removes all nodes after the k-th one. If the tree is double
        recursive, also recomputes the weights to get a double recursive
        subtree.
        """
        n = self.size[0]
        if k > n:
            return self
        else:
            self.limit = k
            new_weight = self.weight[:k]
            if self.is_double_recursive():
                new_weight = standardisation(new_weight)
            new_add = self.additional.copy()
            for j in range(k, n):
                _ = new_add.pop(j, None)
            code = self.convert("code")[: k - 1]
            self.__init__(max_size=k)
            for i in code:
                self.add_node(i)
            self.weight = new_weight
            self.additional = new_add
            return self

    def subtree(
        self, k: int, renormalise_weights: bool = False
    ) -> RecursiveTree:
        """
        Returns the recursive subtree based at k.

        According to the value of the parameter renormalise_weights,
        the weights can be recomputed to be a standardisation of
        the original set of weights.
        """
        from .transformations import tree_subtree

        return tree_subtree(self, k, renormalise_weights)

    def cut(self, k: int, renormalise_weights: bool = False) -> RecursiveTree:
        """
        Removes the subtree based at k (including k), and renormalises the
        labels.

        According to the value of the parameter renormalise_weights,
        the weights can be recomputed to be a standardisation of
        the original set of weights.
        """
        from .transformations import tree_cut

        return tree_cut(self, k, renormalise_weights)

    def add_node(self, k: int, **kwargs) -> Self:
        """
        Adds a node with label (n = size of the tree) as a child of k.

        Additionnally:
        - a function map_weights can be applied to the weights
          before the grafting.
        - the weight of the new node can be specified with the argument
          new_weight.
        """
        from .transformations import tree_add_node

        tree_add_node(self, k, **kwargs)
        return self

    def KP_insertion(self, i: int, J: int) -> Self:
        """
        Realises the Kuba-Panholzer insertion at node i, with a new
        weight equal to J.

        The rules are as follows:
        - if the existing tree has size n,  we add the node with label n
          over the node with label i in [0, n-1].
        - the new node is given the weight J in [1, w(i)].
        - we raise by 1 all the weights j>=J (except for the new node).
        """
        return self.add_node(
            i, map_weights=(lambda x: x + int(x >= J)), new_weight=J
        )

    def KP_insertion_at_weight(self, i: int, J: int) -> Self:
        """
        Realises the Kuba-Panholzer insertion at the node with weight i,
        with a new weight equal to J.
        """
        p = np.argwhere(self.weights == i)[0][0]
        return self.KP_insertion(p, J)

    ##################
    # Random objects #
    ##################

    def random_node(self, with_weights: bool = False) -> int:
        """
        Picks a random node of the tree.

        According to the value of the parameter with_weights, the node i
        can be chosen uniformly, or with probability proportional to the
        weight w[i].
        """
        n = self.size[0]
        if with_weights:
            return rand.choice(n, p=self.weight[:n] / sum(self.weight[:n]))
        else:
            return rand.randint(0, n)

    def random_subtree(self):
        """
        Picks a random node and returns the corresponding recursive subtree.
        """
        from ..random.random_trees import (
            DeterministicRecursiveTree,
            RandomSubtree,
        )

        return RandomSubtree(DeterministicRecursiveTree(self))

    def random_cut(self):
        """
        Picks a random node and returns the corresponding cut.
        """
        from ..random.random_trees import DeterministicRecursiveTree, RandomCut

        return RandomCut(DeterministicRecursiveTree(self))

    def random_leaf(self, subtree: list = []) -> int:
        """
        Picks a leaf at random according to the hook algorithm.

        The optional argument allows one to apply the algorithm to
        a subtree.
        """
        if subtree == []:
            Ind = list(range(self.size[0]))
        else:
            Ind = subtree
        L = [k for k in self.subtree_indices(rand.choice(Ind)) if k in Ind]
        while len(L) > 1:
            r = rand.choice(L[1:])
            L = [k for k in self.subtree_indices(r) if k in Ind]
        return int(L[0])

    def random_relabelling(self) -> None:
        """
        Relabels the nodes of the tree, by choosing uniformly at random
        among all the possible increasing labellings of the underlying
        rooted tree.
        """
        n = self.size[0]
        d = np.zeros(n, dtype=int)
        dinv = np.zeros(n, dtype=int)
        Ind = list(range(n))
        while n > 0:
            k = self.random_leaf(subtree=Ind)
            Ind.remove(k)
            d[n - 1] = k
            dinv[k] = n - 1
            n -= 1
        n = self.size[0]
        self.parent[1:n] = dinv[self.parent[d[1:n]]]
        self.weight[:n] = self.weight[d]
        self.size[:n] = self.size[d]
        self.depth[:n] = self.depth[d]
        new_children = np.empty(n, dtype=object)
        for k in range(n):
            new_children[k] = dinv[self.children[d[k]]].tolist()
        self.children[:n] = new_children
