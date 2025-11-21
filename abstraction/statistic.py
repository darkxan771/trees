import numpy as np
import matplotlib.pyplot as plt

from .tree import Tree


class TreeStatistic:
    """
    A statistic of the nodes of a tree.

    Available statistics are:
    - the degrees of the nodes (name="degree").
    - the depths of the nodes (name="depth").
    - the sizes of the subtrees based at a node (name="size").

    According to the value of the parameter with_weights, the
    distribution can be taken with respect to the uniform law
    on nodes, or with respect to the law proportional to the weights.
    """

    def __init__(self, T: Tree, name: str, with_weights: bool = False):
        self.tree = T
        self.name = name
        self.with_weights = with_weights

    def __repr__(self):
        res = self.name.capitalize() + f" statistic of a {self.tree}"
        if self.with_weights:
            res += " with weights"
        return res

    @property
    def array(self) -> np.ndarray:
        """
        The distribution of the tree statistic.
        """
        n = self.tree.number_of_vertices
        if self.with_weights:
            w = self.tree.weights / np.sum(self.tree.weights)
        else:
            w = np.ones(n) / n
        data = self.tree.data(self.name)
        M = max(data)
        res = np.array([np.sum((data == d) * w) for d in range(M + 1)])
        return res

    @property
    def mean(self) -> float:
        """
        The mean of the tree statistic.
        """
        dist = self.array
        return float(np.sum(dist * np.arange(dist.size)))

    @property
    def var(self) -> float:
        """
        The variance of the tree statistic.
        """
        dist = self.array
        EX2 = np.sum(dist * np.arange(dist.size) ** 2)
        EX = np.sum(dist * np.arange(dist.size))
        return float(EX2 - EX**2)

    def plot(self, limit: int = 10**100) -> None:
        """
        Plots the histogram of the distribution of the tree statistic.
        """
        dist = self.array
        fig, ax = plt.subplots()
        L = min(limit, len(dist))
        ax.bar(np.arange(L), dist[:L])
        ax.set_xticks(np.arange(L))
        S = self.name
        if self.with_weights:
            ax.set_title(f"Weighted distribution of the {S} statistic")
        else:
            ax.set_title(f"Distribution of the {S} statistic")
        plt.show()
