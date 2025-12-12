# Plots of IntegerPartition and Tree objects


from __future__ import annotations

from typing import Callable

import networkx as nx
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.patches import Circle
from networkx import spring_layout

from ..abstraction.helpers import standardisation
from ..recursive import RecursiveTree
from .partition import IntegerPartition
from .tree import Tree

###########
# Layouts #
###########


def angles(T: RecursiveTree) -> np.ndarray:
    n = T.number_of_vertices
    C = T.children
    angle_min = np.zeros(n, dtype=float)
    angle_max = np.zeros(n, dtype=float)
    angle_max[0] = 2 * np.pi
    for i in range(1, n):
        p = T.parent[i]
        j = list(C[p]).index(i)
        denom = T.size[p] - 1
        tmin = sum(T.size[k] for k in C[p][:j]) / denom
        tmax = sum(T.size[k] for k in C[p][: j + 1]) / denom
        if p == 0:
            angle_min[i] = tmin * 2 * np.pi
            angle_max[i] = tmax * 2 * np.pi
        else:
            span = angle_max[p] - angle_min[p]
            angle_min[i] = angle_min[p] + (0.1 + 0.8 * tmin) * span
            angle_max[i] = angle_min[p] + (0.1 + 0.8 * tmax) * span
    return (angle_min + angle_max) / 2


def radii(T: RecursiveTree, style: str = "circular") -> np.ndarray:
    D = T.depth[: T.number_of_vertices]
    if style == "log-circular":
        return np.log(1 + D)
    else:
        return D


def compute_centered(T: RecursiveTree) -> dict:
    D = T.depth
    prof = T.profile
    rpos = T.row_positions
    return {
        i: np.array([-(prof[D[i]] + 1) / 2 + rpos[i], D[i]])
        for i in range(T.size[0])
    }


def compute_left_aligned(T: RecursiveTree) -> dict:
    rpos = T.row_positions
    return {i: np.array([rpos[i], T.depth[i]]) for i in range(T.size[0])}


def compute_circular(
    T: RecursiveTree, rad: np.ndarray, ang: np.ndarray
) -> dict:
    return {
        i: np.array([rad[i] * np.cos(ang[i]), rad[i] * np.sin(ang[i])])
        for i in range(T.size[0])
    }


def compute_natural(T: RecursiveTree) -> dict:
    Tn = T.convert("networkx")
    pos0 = compute_circular(T, radii(T), angles(T))
    return spring_layout(Tn, pos=pos0, k=0.1, iterations=300)


compute_layouts: dict[str, Callable] = {
    "centered": compute_centered,
    "left-aligned": compute_left_aligned,
    "circular": lambda T: compute_circular(T, radii(T), angles(T)),
    "log-circular": lambda T: compute_circular(
        T, radii(T, "log-circular"), angles(T)
    ),
    "natural": compute_natural,
}


###################
# Labels / colors #
###################


def labels_simple(T: RecursiveTree) -> dict:
    return {i: str(i) for i in range(T.size[0])}


def labels_double(T: RecursiveTree) -> dict:
    L = labels_simple(T)
    for i in range(T.size[0]):
        L[i] += "|" + str(int(T.size[0] - T.weight[i]))
    return L


def labels_with_weights(T: RecursiveTree) -> dict:
    L = labels_simple(T)
    for i in range(T.size[0]):
        L[i] += ":" + str(int(T.weight[i]))
    return L


compute_labels: dict[str, Callable] = {
    "simple": labels_simple,
    "double": labels_double,
    "with_weights": labels_with_weights,
}


color_start = np.array([0, 70, 135]) / 255
color_end = np.array([255, 68, 153]) / 255


class ComputeColors:

    def __init__(self, opt: str):
        self.opt = opt

    def __call__(
        self, T: RecursiveTree, G: GraphicOptions
    ) -> tuple[list, list]:
        n = T.number_of_vertices
        E = T.convert("networkx").edges
        f = colors_dict[self.opt](T, G)
        return [f(k) for k in range(n)], [f(e[1]) for e in E]


def colors_age(
    T: RecursiveTree, _: GraphicOptions
) -> Callable[[int], np.ndarray]:
    birth = np.array(list(T.birth_times.values()))
    bs = standardisation(birth) / T.number_of_vertices
    return lambda x: (1 - bs[x]) * color_start + bs[x] * color_end


def colors_trim(
    T: RecursiveTree, G: GraphicOptions
) -> Callable[[int], np.ndarray]:
    epsilon = G.options["trim"]
    start = T.trim_indices(epsilon)
    return lambda x: color_start if x in start else color_end


def colors_subtree(
    T: RecursiveTree, G: GraphicOptions
) -> Callable[[int], np.ndarray]:
    k = G.options["subtree"]
    end = T.subtree_indices(k)
    res = lambda x: color_end if x in end else color_start
    return res


def colors_time(
    T: RecursiveTree, G: GraphicOptions
) -> Callable[[int], np.ndarray]:
    t = G.options["time"]
    Tb = T.birth_times
    return lambda x: color_end if Tb[x] > t else color_start


def colors_longest_path(
    T: RecursiveTree, _: GraphicOptions
) -> Callable[[int], np.ndarray]:
    L = T.longest_path
    return lambda x: color_end if x in L else color_start


colors_dict: dict[str, Callable] = {
    "age": colors_age,
    "trim": colors_trim,
    "subtree": colors_subtree,
    "time": colors_time,
    "longest_path": colors_longest_path,
}


###########
# Options #
###########

graphic_options = """
        node_size (int, 300)
        node_shape (str, "o")
        arrows (bool, True)
        arrow_size (int, 10)
        width (float, 1.0)
        node_color, edge_color
        font_size (int, 12)
        """


class GraphicOptions:
    """
    Set of graphic options for the drawing of trees.
    """

    def __init__(self, dict: dict):
        self.options = dict
        self.style = self.options.pop("style", "centered")
        self.labels = self.options.pop("labels", "simple")
        self.with_levels = self.options.pop("with_levels", False)
        self.with_colors = self.options.pop("with_colors", False)
        if self.options.pop("large", False):
            self.labels = "empty"
            self.options["arrows"] = False
            self.options["node_size"] = 0

    def draw_levels_on_ax(self, ax0: Axes, H: int, prof: np.ndarray) -> None:
        col = (0.8, 0.8, 0.8)
        for i in range(1, H + 1):
            if self.style == "circular":
                ax0.add_patch(Circle((0, 0), i, fill=False, edgecolor=col))
            if self.style == "log-circular":
                ax0.add_patch(
                    Circle((0, 0), np.log(1 + i), fill=False, edgecolor=col)
                )
            if self.style == "centered":
                p = (prof[i] - 1) / 2
                ax0.plot([-p - 1, p - 1], [i, i], color=col)
            if self.style == "left-aligned":
                ax0.plot([0, prof[i] - 1], [i, i], color=col)


##################
# Drawing a tree #
##################


def draw_tree_on_ax(T: Tree, ax0: Axes, **options) -> None:
    G = GraphicOptions(options)
    if T.type == "rooted":
        G.labels = "empty"
    RT = T.convert("recursive")
    Tn = T.convert("networkx")
    ax0.set_axis_off()
    if G.with_levels:
        G.draw_levels_on_ax(ax0, RT.height, RT.profile)
    if G.with_colors:
        nc, ec = ComputeColors(G.with_colors)(RT, G)
        _ = G.options.pop(G.with_colors, None)
    else:
        nc, ec = "b", "k"
    pos0 = compute_layouts[G.style](RT)
    if G.labels == "empty":
        nx.draw_networkx(
            Tn,
            ax=ax0,
            pos=pos0,
            with_labels=False,
            node_color=nc,
            edge_color=ec,
            **G.options,
        )
    else:
        L = compute_labels[G.labels](RT)
        nx.draw_networkx(
            Tn,
            ax=ax0,
            pos=pos0,
            labels=L,
            node_color=nc,
            edge_color=ec,
            **G.options,
        )
    if G.style in ["circular", "log-circular", "natural"]:
        ax0.set_aspect(1)


#######################
# Drawing a partition #
#######################


def draw_partition_on_ax(
    P: IntegerPartition, ax0: Axes, style: str = "french"
) -> None:
    ax0.set_axis_off()
    ax0.set_aspect(1)
    if style == "french":
        ax0.plot([0, 0, P.parts[0]], [P.length, 0, 0], color="k")
        for j, p in enumerate(P.parts):
            for i in range(p):
                ax0.plot([i + 1, i + 1, i], [j, j + 1, j + 1], color="k")
    if style == "english":
        ax0.plot([0, 0, P.parts[0]], [-P.length, 0, 0], color="k")
        for j, p in enumerate(P.parts):
            for i in range(p):
                ax0.plot([i + 1, i + 1, i], [-j, -j - 1, -j - 1], color="k")
    if style == "russian":
        ax0.plot(
            [-P.length, 0, P.parts[0]], [P.length, 0, P.parts[0]], color="k"
        )
        for j, p in enumerate(P.parts):
            for i in range(p):
                ax0.plot(
                    [i + 1 - j, i - j, i - j - 1],
                    [i + 1 + j, i + 2 + j, i + 1 + j],
                    color="k",
                )
