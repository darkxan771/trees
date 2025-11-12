import numpy as np

from typing import Callable
from networkx import spring_layout
from .recursive_trees import RecursiveTree


def angles(T: RecursiveTree) -> np.ndarray:
    n = T.size[0]
    angle_min = np.zeros(n, dtype=float)
    angle_max = np.zeros(n, dtype=float)
    angle_max[0] = 2 * np.pi
    for i in range(1, n):
        p = T.parent[i]
        j = list(T.children[p]).index(i)
        denom = T.size[p] - 1
        tmin = sum(T.size[k] for k in T.children[p][:j]) / denom
        tmax = sum(T.size[k] for k in T.children[p][: j + 1]) / denom
        if p == 0:
            angle_min[i] = tmin * 2 * np.pi
            angle_max[i] = tmax * 2 * np.pi
        else:
            span = angle_max[p] - angle_min[p]
            angle_min[i] = angle_min[p] + (0.1 + 0.8 * tmin) * span
            angle_max[i] = angle_min[p] + (0.1 + 0.8 * tmax) * span
    return (angle_min + angle_max) / 2


def radii(T: RecursiveTree, style: str = "circular") -> np.ndarray:
    D = T.depth[: T.size[0]]
    if style == "log-circular":
        return np.log(1 + D)
    else:
        return D


def compute_centered(T: RecursiveTree) -> dict:
    D = T.depth
    prof = T.profile()
    rpos = T.row_positions()
    return {
        i: np.array([-(prof[D[i]] + 1) / 2 + rpos[i], D[i]])
        for i in range(T.size[0])
    }


def compute_left_aligned(T: RecursiveTree) -> dict:
    rpos = T.row_positions()
    return {i: np.array([rpos[i], T.depth[i]]) for i in range(T.size[0])}


def compute_circular(
    T: RecursiveTree, rad: np.ndarray, ang: np.ndarray
) -> dict:
    return {
        i: np.array([rad[i] * np.cos(ang[i]), rad[i] * np.sin(ang[i])])
        for i in range(T.size[0])
    }


def compute_natural(T: RecursiveTree) -> dict:
    Tn = T.to_networkx()
    pos0 = compute_circular(T, radii(T), angles(T))
    return spring_layout(Tn, pos=pos0, k=0.1, iterations=300)


compute_layouts: dict[str, Callable] = {
    "centered": compute_centered,
    "left-aligned": compute_left_aligned,
    "circular": (lambda T: compute_circular(T, radii(T), angles(T))),
    "log-circular": (
        lambda T: compute_circular(T, radii(T, "log-circular"), angles(T))
    ),
    "natural": compute_natural,
}


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
