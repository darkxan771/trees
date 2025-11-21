import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Protocol, Any, Callable
from matplotlib.axes._axes import Axes
from .partition import IntegerPartition

sns.set_theme()


class Tree(Protocol):
    @property
    def type(self) -> str:
        ...

    @property
    def number_of_vertices(self) -> int:
        ...

    @property
    def number_of_edges(self) -> int:
        return self.number_of_vertices - 1

    @property
    def height(self) -> int:
        ...

    @property
    def profile(self) -> np.ndarray:
        ...

    @property
    def subtrees_partition(self) -> IntegerPartition:
        ...

    @property
    def weights(self) -> np.ndarray:
        ...

    @property
    def convert(self) -> Callable[[str], Any]:
        ...

    @property
    def data(self) -> Callable[[str], np.ndarray]:
        ...

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
        """
        fig, ax0 = plt.subplots(figsize=(8, 8))
        self.draw_on_ax(ax0, **options)
        plt.show()
