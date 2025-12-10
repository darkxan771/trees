# Defines the abstract Tree class

from typing import Any, Callable, Protocol

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes._axes import Axes

from .partition import IntegerPartition

sns.set_theme()


class Tree(Protocol):
    @property
    def type(self) -> str: ...

    @property
    def number_of_vertices(self) -> int: ...

    @property
    def number_of_edges(self) -> int:
        return self.number_of_vertices - 1

    @property
    def height(self) -> int: ...

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
        res = [T.number_of_vertices for T in self.subtree_list]
        res.sort(reverse=True)
        return IntegerPartition(res)

    @property
    def weights(self) -> np.ndarray: ...

    @property
    def convert(self) -> Callable[[str], Any]: ...

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
