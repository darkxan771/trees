from __future__ import annotations
import numpy as np
import numpy.random as rand
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from typing import Callable, TYPE_CHECKING
from matplotlib.patches import Circle
from matplotlib.axes._axes import Axes
from .conversions import standardisation, code_to_permutation

sns.set_theme()

if TYPE_CHECKING:
    from .rooted_trees import RootedTree


extract_statistic: dict[str, Callable] = {
    "degree": (lambda T: T.degrees()),
    "depth": (lambda T: T.depth[: T.size[0]]),
    "size": (lambda T: T.size[: T.size[0]]),
}


class RecursiveTree:
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
        return hash(self.to_code())

    def __eq__(self, other):
        A = isinstance(other, RecursiveTree)
        B = self.to_code() == other.to_code()
        return A and B

    def first_columns(self, d: int = 0) -> pd.DataFrame:
        """
        Returns the d first columns of the array encoding the recursive tree.
        """
        if d == 0:
            D = self.size[0]
        else:
            D = d
        matrix = np.array(
            [
                self.parent[:D],
                self.weight[:D],
                self.size[:D],
                self.depth[:D],
                self.children[:D],
            ]
        )
        row_names = np.array(["parent", "weight", "size", "depth", "children"])
        column_names = np.arange(D)
        return pd.DataFrame(matrix, columns=column_names, index=row_names)

    ###############
    # Conversions #
    ###############

    def to_code(self) -> tuple:
        """
        Returns the code of the recursive tree (list of the nodes of
        attachment).
        """
        return tuple(self.parent[1:].tolist())

    def to_permutation(self) -> tuple:
        """
        Returns the permutation of the recursive tree, obtained via one of
        the bijection from recursive trees with size n to permutations with
        size n-1.
        """
        return tuple((code_to_permutation(self.parent[1:])).tolist())

    def to_networkx(self) -> nx.classes.digraph.DiGraph:
        """
        Encodes the tree in a NetworkX labelled digraph.
        """
        T = nx.DiGraph({i: self.children[i] for i in range(self.size[0])})
        for i in range(self.size[0]):
            T.nodes[i]["label"] = i
            T.nodes[i]["depth"] = self.depth[i]
            T.nodes[i]["weight"] = self.weight[i]
        return T

    def to_rooted_tree(self) -> RootedTree:
        """
        Forgets the labels and weights and returns the rooted tree
        """
        from .rooted_trees import RootedTree

        subs = self.children[0]
        return RootedTree([self.subtree(c).to_rooted_tree() for c in subs])

    ###################################
    # Extract statistical information #
    ###################################

    def number_of_edges(self) -> int:
        """
        Returns the number of edges of the tree.
        """
        return int(self.size[0] - 1)

    def number_of_vertices(self) -> int:
        """
        Returns the size of the tree (number of vertices).
        """
        return int(self.size[0])

    def height(self) -> int:
        """
        Returns the height of the tree (maximal depth of a node).
        """
        return int(np.max(self.depth))

    def profile(self) -> np.ndarray:
        """
        Returns the profile of the tree (number of nodes on each level).
        """
        h = self.height()
        return np.array(
            [np.count_nonzero(self.depth == d) for d in range(h + 1)]
        )

    def degrees(self) -> np.ndarray:
        """
        Returns the degrees of the nodes of the tree.
        """
        return np.vectorize(lambda L: len(L))(self.children[: self.size[0]])

    def path_to_root(self, k: int) -> np.ndarray:
        """
        Returns the unique path from the root to k.
        """
        res = np.zeros(self.depth[k] + 1, dtype=int)
        res[-1] = k
        for i in range(self.depth[k]):
            res[-i - 2] = self.parent[res[-i - 1]]
        return res

    def subtree_indices(self, k: int) -> list[int]:
        """
        Returns the set of indices in the subtree based at k.
        """
        L = sum([self.subtree_indices(int(n)) for n in self.children[k]], [k])
        L.sort()
        return L

    def row_positions(self) -> np.ndarray:
        """
        Returns an array with the row positions of the nodes.
        """
        res = np.zeros(self.size[0], dtype=int)
        count = np.zeros(self.height(), dtype=int)
        h = self.height()
        level = [0]
        next_level = self.children[0]
        for d in range(h):
            for x in next_level:
                res[x] = count[d]
                count[d] += 1
            level = next_level
            next_level = sum([self.children[x] for x in level], [])
        return res

    def distribution(
        self, statistic: str = "degree", with_weights: bool = False
    ) -> np.ndarray:
        """
        Computes the distribution of a statistic of the nodes of the tree.

        Available statistics are:
        - the degree of a node (statistic="degree").
        - the depth of a node (statistic="depth").
        - the size of the subtree based at a node (statistic="size").

        According to the value of the parameter with_weights, the
        distribution can be computed with respect to the uniform law
        on nodes, or with respect to the law proportional to the weights.
        """
        n = self.size[0]
        if with_weights:
            w = self.weight[:n] / np.sum(self.weight[:n])
        else:
            w = np.ones(n) / n
        data = extract_statistic[statistic](self)
        M = max(data)
        res = np.zeros(M + 1, dtype=float)
        for d in range(M + 1):
            res[d] = np.sum((data == d) * w)
        return res

    def mean(
        self, statistic: str = "degree", with_weights: bool = False
    ) -> float:
        """
        Computes the mean of a statistic of the nodes of the tree.
        """
        dist = self.distribution(statistic, with_weights)
        return float(np.sum(dist * np.arange(dist.size)))

    def var(
        self, statistic: str = "degree", with_weights: bool = False
    ) -> float:
        """
        Computes the variance of a statistic of the nodes of the tree.
        """
        dist = self.distribution(statistic, with_weights)
        EX2 = np.sum(dist * np.arange(dist.size) ** 2)
        EX = np.sum(dist * np.arange(dist.size))
        return float(EX2 - EX**2)

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

    def extract_KP_insertion_array(self) -> np.ndarray:
        """
        If the recursive tree is double recursive, returns the
        corresponding Kuba-Panholzer insertion array.
        """
        if not self.is_double_recursive():
            raise ValueError("The tree is not double recursive.")
        else:
            n = self.size[0]
            res = np.zeros((2, n - 1), dtype=int)
            W = self.weight[:n].copy()
            for i in range(1, n):
                J = W[n - i]
                res[:, n - i - 1] = np.array([W[self.parent[n - i]] - 1, J])
                W[: n - i] -= W[: n - i] >= J
            return res

    ##############################
    # Modifications / insertions #
    ##############################

    def map_weights(self, func) -> None:
        """
        Maps a function on the set of weights.
        """
        n = self.size[0]
        self.weight[:n] = np.vectorize(func)(self.weight[:n])

    def add_node(self, parent_label: int, **kwargs) -> None:
        """
        Adds a node with label (n = size of the tree) as a child of
        (i = parent_label).

        Additionnally:
        - a function map_weights can be applied to the weights
          before the grafting.
        - the weight of the new node can be specified with the argument
          new_weight.
        """
        n = self.size[0]
        self.parent[n] = parent_label
        self.depth[n] = self.depth[parent_label] + 1
        self.children[n] = []
        self.children[parent_label] += [n]
        for k in self.path_to_root(n):
            self.size[k] += 1
        if "map_weights" in kwargs:
            self.map_weights(kwargs["map_weights"])
        if "new_weight" in kwargs:
            self.weight[n] = kwargs["new_weight"]
        else:
            self.weight[n] = 1

    def KP_insertion(self, i: int, J: int) -> None:
        """
        Realises the Kuba-Panholzer insertion at node i, with a new
        weight equal to J.

        The rules are as follows:
        - if the existing tree has size n,  we add the node with label n
          over the node with label i in [0, n-1].
        - the new node is given the weight J in [1, w(i)].
        - we raise by 1 all the weights j>=J (except for the new node).
        """
        self.add_node(i, map_weights=(lambda x: x + int(x >= J)), new_weight=J)

    def KP_insertion_at_weight(self, i: int, J: int) -> None:
        """
        Realises the Kuba-Panholzer insertion at the node with weight i,
        with a new weight equal to J.
        """
        n = self.size[0]
        p = np.argwhere(self.weight[:n] == i)[0][0]
        self.KP_insertion(p, J)

    def subtree(
        self, k: int, normalise_weights: bool = False
    ) -> "RecursiveTree":
        """
        Returns the recursive subtree based at k.

        According to the value of the parameter normalise_weights,
        the weights can be recomputed to be a standardisation of
        the original set of weights.
        """
        sub = np.array(self.subtree_indices(k))
        T = RecursiveTree(max_size=int(self.size[k]))
        dict_sub = {sub[i]: i for i in range(self.size[k])}
        for i in range(self.size[k]):
            if i > 0:
                T.parent[i] = dict_sub[self.parent[sub[i]]]
            else:
                T.parent[i] = -1
            T.children[i] = [dict_sub[int(n)] for n in self.children[sub[i]]]
        T.size = self.size[sub]
        T.depth = self.depth[sub] - self.depth[k]
        T.weight = self.weight[sub]
        if normalise_weights:
            T.weight = standardisation(T.weight)
        return T

    def cut(self, k: int, normalise_weights: bool = False) -> "RecursiveTree":
        """
        Removes the subtree based at k, and renormalises the labels.

        According to the value of the parameter normalise_weights,
        the weights can be recomputed to be a standardisation of
        the original set of weights.
        """
        n = self.size[0]
        sub = self.subtree_indices(k)
        sub.remove(k)
        to_substrack = len(sub)
        keep = [i for i in range(n) if (not (i in sub))]
        dict_keep = {keep[i]: i for i in range(len(keep))}
        dict_keep[-1] = -1
        T = RecursiveTree(max_size=len(keep))
        T.parent = self.parent[keep]
        T.parent = np.vectorize(lambda i: dict_keep[i])(T.parent)
        T.children = self.children[keep]
        T.children[dict_keep[k]] = []
        for i in range(len(keep)):
            T.children[i] = list(map(lambda i: dict_keep[i], T.children[i]))
        T.size = self.size[keep]
        for i in self.path_to_root(k):
            T.size[dict_keep[i]] -= to_substrack
        T.depth = self.depth[keep]
        T.weight = self.weight[keep]
        if normalise_weights:
            T.weight = standardisation(T.weight)
        return T

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

    def random_subtree(
        self, with_weights: bool = False, normalise_weights: bool = False
    ) -> "RecursiveTree":
        """
        Picks a random node and returns the corresponding recursive subtree.

        The parameter with_weights decides how the node is chosen randomly,
        and the parameter normalise_weights, when set to True, replaces the
        weights of the subtree by a standardisation of this set.
        """
        k = self.random_node(with_weights)
        return self.subtree(k, normalise_weights)

    def random_cut(
        self, with_weights: bool = False, normalise_weights: bool = False
    ) -> "RecursiveTree":
        """
        Picks a random node and returns the corresponding cut.

        The parameter with_weights decides how the node is chosen randomly,
        and the parameter normalise_weights, when set to True, replaces the
        weights of the subtree by a standardisation of this set.
        """
        k = self.random_node(with_weights)
        return self.cut(k, normalise_weights)

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

    #################
    # Visualisation #
    #################

    def plot_distribution(
        self,
        statistic: str = "degree",
        with_weights: bool = False,
        limit: int = 10**100,
    ) -> None:
        """
        Plots the histogram of the distribution of a statistic of the nodes.
        """
        dist = self.distribution(statistic, with_weights)
        fig, ax = plt.subplots()
        L = min(limit, len(dist))
        ax.bar(np.arange(L), dist[:L])
        ax.set_xticks(np.arange(L))
        S = statistic
        if with_weights:
            ax.set_title(f"Weighted distribution of the {S} of a random node")
        else:
            ax.set_title(f"Distribution of the {S} of a random node")
        plt.show()

    def draw_on_ax(
        self,
        ax0: Axes,
        style: str = "centered",
        labels: str = "simple",
        with_circles: bool = False,
        **kwargs,
    ) -> None:
        """
        Draws the tree on a Matplotlib Axes object.
        """
        from .draw_helpers import compute_layouts, compute_labels

        T = self.to_networkx()
        ax0.set_axis_off()
        if with_circles:
            edge_c = (0.8, 0.8, 0.8)
            for i in range(self.height() + 1):
                ax0.add_patch(Circle((0, 0), i, fill=False, edgecolor=edge_c))
        n = self.size[0]
        pos0 = compute_layouts[style](self)
        if labels == "empty":
            nx.draw_networkx(T, ax=ax0, pos=pos0, with_labels=False, **kwargs)
        else:
            L = compute_labels[labels](self)
            nx.draw_networkx(T, ax=ax0, pos=pos0, labels=L, **kwargs)

    def plot(
        self,
        style: str = "centered",
        labels: str = "simple",
        with_circles: bool = False,
        **kwargs,
    ) -> None:
        """
        Plots the recursive tree.

        Available options:

        - style: "centered", "left-aligned", "natural",
                  "circular", "log-circular".
        - labels: "empty", "simple", "with_weights", "double".
        - with_circles: bool.
        """
        fig, ax0 = plt.subplots(figsize=(8, 8))
        if style in ["circular", "log-circular", "natural"]:
            ax0.set_aspect(1)
        self.draw_on_ax(ax0, style, labels, with_circles, **kwargs)
        plt.show()
