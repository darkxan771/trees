# Defines the class RecursiveTree


from __future__ import annotations

from collections.abc import Sequence
from typing import Callable
from typing import Self

import numpy as np
import numpy.random as rand

from ..abstraction import IntegerPartition
from ..abstraction import Tree

compute_data: dict[str, Callable] = {
    "degree": lambda T: np.vectorize(lambda L: len(L))(
        np.array(T.children, dtype=object)
    ),
    "depth": lambda T: T.depth[: T.size],
    "size": lambda T: T.n[: T.size],
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
    - T.n[i] is the size of the subtree based at i. In particular,
      T.n[0] = n.
    - T.depth[i] is the depth of the node i (distance to the root). In
      particular, T.depth[0] = 0.

    For branching processes, the birth times and birth_processes
    are saved in T.birth_times and T.birth_processes.
    """

    def __init__(self, max_size: int = 10**6):
        self.parent = -np.ones(max_size, dtype=int)
        self.weight = np.zeros(max_size, dtype=int)
        self.weight[0] = 1
        self.n = np.zeros(max_size, dtype=int)
        self.n[0] = 1
        self.depth = -np.ones(max_size, dtype=int)
        self.depth[0] = 0
        self.limit = max_size
        self.birth_times: dict[int, float] = {}
        self.birth_processes: dict = {}

    def __abs__(self) -> int:
        return int(self.n[0])

    def __len__(self) -> int:
        return int(np.max(self.depth))

    def copy(self, new_size: int | None = None) -> RecursiveTree:
        """
        Copy to a new recursive tree. The additional parameter new_size
        allows one to resize the tree.
        """
        if new_size is None:
            L = self.limit
        else:
            L = new_size
        m = min([self.limit, L])
        code = self.convert("code")[: m - 1]
        R = RecursiveTree(L)
        for i in code:
            R.add_node(i)
        n = len(list(self.birth_times.keys()))
        R.weight[:m] = self.weight[:m].copy()
        R.birth_times = self.birth_times.copy()
        R.birth_processes = self.birth_processes.copy()
        for k in range(n):
            if k >= m:
                _ = R.birth_times.pop(k, None)
                _ = R.birth_processes.pop(k, None)
        return R

    @staticmethod
    def from_code(c: Sequence[int]) -> RecursiveTree:
        """
        Constructs the unique recursive tree corresponding to
        the code c.
        """
        res = RecursiveTree()
        res.weight[0] = len(c) + 1
        for k in range(len(c)):
            _ = res.add_node(c[k], new_weight=len(c) - k)
        return res

    @classmethod
    def from_permutation(cls, p: Sequence[int]) -> RecursiveTree:
        """
        Constructs the unique recursive tree corresponding to
        the permutation p.
        """
        from ..abstraction.conversions import permutation_to_code

        v = np.array(p)
        v.sort()
        if not np.all(v == np.arange(len(v))):
            raise ValueError("The parameter p is not a permutation")
        return cls.from_code(permutation_to_code(np.array(p)).tolist())

    @staticmethod
    def from_KP_insertion_array(L: np.ndarray) -> RecursiveTree:
        """
        Constructs the unique double recursive tree corresponding to
        the insertion array L.

        Each column of L is a pair (i, J) corresponding to the insertion
        of a new node with weight J above the node with weight i.
        """
        T = RecursiveTree(max_size=L.shape[1] + 1)
        for v in L.transpose()[:,]:
            _ = T.KP_insertion_at_weight(v[0], v[1])
        return T

    ##############
    # Properties #
    ##############

    @property
    def category(self) -> str:
        return "recursive tree"

    @property
    def profile(self) -> np.ndarray:
        """
        The profile of the tree (number of nodes on each level).
        """
        h = self.height
        return np.array(
            [np.count_nonzero(self.depth == d) for d in range(h + 1)]
        )

    def children_of_node(self, k) -> list[int]:
        """
        Returns the list of the children of the node k.
        """
        n = self.size
        return [x for x in range(k + 1, n) if self.parent[x] == k]

    @property
    def children(self) -> list[list[int]]:
        """
        A list of lists of the children of each node of the tree.
        """
        n = self.size
        return [self.children_of_node(k) for k in range(n)]

    @property
    def subtrees_partition(self) -> IntegerPartition:
        """
        The integer partition with size n-1 corresponding to the
        sizes of the subtrees attached to the root.
        """
        children = np.array(self.children_of_node(0))
        res = self.n[children].tolist()
        res.sort(reverse=True)
        return IntegerPartition(res)

    @property
    def weights(self) -> np.ndarray:
        """
        The weights of the nodes of the recursive tree.
        """
        return self.weight[: self.size]

    @property
    def data(self) -> Callable[[str], np.ndarray]:
        return lambda name: compute_data[name](self)

    @property
    def row_positions(self) -> np.ndarray:
        """
        The row positions of the nodes.
        """
        children = self.children
        res = np.zeros(self.size, dtype=int)
        h = self.height
        count = np.zeros(h, dtype=int)
        level = [0]
        next_level = children[0]
        for d in range(h):
            for x in next_level:
                res[x] = count[d]
                count[d] += 1
            level = next_level
            next_level = sum([children[x] for x in level], [])
        return res

    @property
    def longest_path(self) -> list[int]:
        """
        Returns the longest path in the tree. If there are several such paths,
        then the longest path is the one with the largest label of final leaf.
        """
        h = self.height
        n = self.size
        candidates = [k for k in range(n) if self.depth[k] == h]
        final = max(candidates)
        return self.path_to_root(final)

    def has_recursive_part(self, L) -> bool:
        """
        Checks if a list of indices forms a recursive part of the
        tree (then, the complement can be removed with the cut method).
        """
        return all([all([i in L for i in self.path_to_root(k)]) for k in L])

    def subtree_indices(self, k: int) -> list[int]:
        """
        Returns the set of indices in the subtree based at k.
        """
        n = self.size
        L = [k]
        to_be_computed = [k]
        while len(to_be_computed) > 0:
            l = to_be_computed.pop(0)
            toadd = [x for x in range(l + 1, n) if self.parent[x] == l]
            to_be_computed += toadd
            L += toadd
        L.sort()
        return L

    def trim_indices(self, epsilon: float) -> list[int]:
        """
        Returns the list of indices of nodes with relative size of
        subtree greater than epsilon, and such that all the ancestors
        have the same property.
        """
        n = self.size
        res = [
            k
            for k in range(n)
            if k == 0 or self.n[k] >= epsilon * self.n[self.parent[k]]
        ]
        return [k for k in res if all([l in res for l in self.path_to_root(k)])]

    def path_to_root(self, k: int) -> list:
        """
        Returns the unique path from the root to k.
        """
        res = np.zeros(self.depth[k] + 1, dtype=int)
        res[-1] = k
        for i in range(self.depth[k]):
            res[-i - 2] = self.parent[res[-i - 1]]
        return res.tolist()

    def is_double_recursive(self) -> bool:
        """
        Checks whether the tree is double recursive (i.e.,
        its set of weights is [1, n] and the weights are
        decreasing.
        """
        n = self.size
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
        n = self.size
        self.weight[:n] = np.vectorize(func)(self.weight[:n])
        return self

    def resize(self, k: int) -> RecursiveTree:
        """
        Removes all nodes after the k-th one.
        """
        if k <= self.size:
            return self.copy(k)
        else:
            raise ValueError("One cannot resize the tree to a larger size.")

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

    @property
    def subtree_list(self) -> list:
        """
        The list of subtrees of the tree.
        """
        return [self.subtree(k) for k in self.children_of_node(0)]

    def cut(
        self, L: list[int], renormalise_weights: bool = False
    ) -> RecursiveTree:
        """
        Removes all the nodes of L if the complement is a recursive part.

        According to the value of the parameter renormalise_weights,
        the weights can be recomputed to be a standardisation of
        the original set of weights.
        """
        from .transformations import tree_cut

        return tree_cut(self, L, renormalise_weights)

    def trim(self, epsilon: float) -> RecursiveTree:
        """
        Removes all the subtrees with relative size smaller than epsilon.
        """
        from .transformations import tree_cut

        trim_indices = self.trim_indices(epsilon)
        n = self.size
        to_cut = [k for k in range(n) if not (k in trim_indices)]
        return tree_cut(self, to_cut)

    def add_node(self, k: int, **kwargs) -> Self:
        """
        Adds a node with label (n = size of the tree) as a child of k.

        Additionnally:
        - a function map_weights can be applied to the weights
          before the grafting.
        - the weight of the new node can be specified with the argument
          new_weight.
        - the birth time and birth process of the new node can be specified.
        """
        from .transformations import tree_add_node

        tree_add_node(self, k, **kwargs)
        return self

    def insert_node(self, l: int, parent: int, **kwargs) -> Self:
        """
        Shifts all the indices greater than l by one, and
        inserts then a l-th node above the parent.

        The birth time and birth process of the new l-th node can be specified.
        """
        from .transformations import tree_insert_node

        tree_insert_node(self, l, parent, **kwargs)
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
        n = self.size
        if with_weights:
            return rand.choice(n, p=self.weight[:n] / sum(self.weight[:n]))
        else:
            return rand.randint(0, n)

    def random_subtree(self):
        """
        Picks a random node and returns the corresponding recursive subtree.
        """
        from ..random.random_trees import DeterministicRecursiveTree
        from ..random.random_trees import RandomSubtree

        return RandomSubtree(DeterministicRecursiveTree(self))

    def random_cut(self):
        """
        Picks a random node and returns the corresponding cut.
        """
        from ..random.random_trees import DeterministicRecursiveTree
        from ..random.random_trees import RandomCut

        return RandomCut(DeterministicRecursiveTree(self))

    def random_leaf(self, subtree: list = []) -> int:
        """
        Picks a leaf at random according to the hook algorithm.

        The optional argument allows one to apply the algorithm to
        a subtree.
        """
        if subtree == []:
            Ind = list(range(self.size))
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
        n = self.size
        d = np.zeros(n, dtype=int)
        dinv = np.zeros(n, dtype=int)
        Ind = list(range(n))
        while n > 0:
            k = self.random_leaf(subtree=Ind)
            Ind.remove(k)
            d[n - 1] = k
            dinv[k] = n - 1
            n -= 1
        n = self.size
        self.parent[1:n] = dinv[self.parent[d[1:n]]]
        self.weight[:n] = self.weight[d]
        self.n[:n] = self.n[d]
        self.depth[:n] = self.depth[d]
