# Defines the abstract Tree class

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes._axes import Axes

from .object import CombinatorialObject
from .partition import IntegerPartition

sns.set_theme()


class Tree(CombinatorialObject):

    def __abs__(self) -> int: ...

    def __len__(self) -> int: ...

    @property
    def category(self) -> str: ...

    @property
    def number_of_vertices(self) -> int:
        """
        The number of vertices of the tree.
        """
        return abs(self)

    @property
    def number_of_edges(self) -> int:
        return abs(self) - 1

    @property
    def height(self) -> int:
        """
        The height of the tree (maximal depth of a node).
        """
        return len(self)

    @property
    def profile(self) -> np.ndarray: ...

    @property
    def subtree_list(self) -> list: ...

    @property
    def subtrees_partition(self) -> IntegerPartition:
        """
        The integer partition with size n-1 corresponding to the
        sizes of the subtrees attached to the root.
        """
        res = [T.size for T in self.subtree_list]
        res.sort(reverse=True)
        return IntegerPartition(res)

    @property
    def weights(self) -> np.ndarray: ...

    @property
    def data(self) -> Callable[[str], np.ndarray]: ...

    def statistic(self, name: str):
        from .statistic import TreeStatistic

        return TreeStatistic(self, name)

    def draw_on_ax(
        self,
        ax0: Axes,
        **options,
    ) -> None:
        """
        Draws the tree on a Matplotlib Axes object.
        """
        from .plot import draw_tree_on_ax

        draw_tree_on_ax(self, ax0, **options)

    def plot(self, **options) -> None:
        """
        Plots the tree.

        Available options:

        - style: "centered", "left-aligned", "natural",
                  "circular", "log-circular".
        - labels: "empty", "simple", "with_weights", "double".
        - with_levels: bool.
        - with_colors: False or "age", "trim", "subtree", "time".
        The three last options need an additional parameter, and they color
        certain parts of the trees in blue or pink. The option "age" only
        works for a tree constructed by a branching process.
        """
        _, ax0 = plt.subplots(figsize=(8, 8))
        self.draw_on_ax(ax0, **options)
        plt.show()
