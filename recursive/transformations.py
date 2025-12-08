# Useful functions for the transformation of recursive trees

import numpy as np

from ..abstraction.helpers import (
    standardisation,
)
from .recursive_tree import RecursiveTree


def tree_subtree(
    T: RecursiveTree, k: int, renormalise_weights: bool = False
) -> RecursiveTree:
    """
    Subtree of T based at k.
    """
    sub = np.array(T.subtree_indices(k))
    S = RecursiveTree(max_size=int(T.size[k]))
    dict_sub = {sub[i]: i for i in range(T.size[k])}
    for i in range(T.size[k]):
        if i > 0:
            S.parent[i] = dict_sub[T.parent[sub[i]]]
        else:
            S.parent[i] = -1
        S.children[i] = [dict_sub[int(n)] for n in T.children[sub[i]]]
    S.size = T.size[sub].copy()
    S.depth = T.depth[sub].copy() - T.depth[k]
    S.weight = T.weight[sub].copy()
    if renormalise_weights:
        S.weight = standardisation(S.weight)
    return S


def tree_cut(
    T: RecursiveTree, k: int, renormalise_weights: bool = False
) -> RecursiveTree:
    """
    Cuts the subtree based at k.
    """
    n = T.number_of_vertices
    sub = T.subtree_indices(k)
    to_substrack = len(sub)
    keep = [i for i in range(n) if (not (i in sub))]
    dict_keep = {keep[i]: i for i in range(len(keep))}
    dict_keep[-1] = -1
    C = RecursiveTree(max_size=len(keep))
    C.parent = T.parent[keep].copy()
    C.parent = np.vectorize(lambda i: dict_keep[i])(C.parent)
    C.children = T.children[keep].copy()
    C.children[dict_keep[T.parent[k]]].remove(k)
    for i in range(len(keep)):
        C.children[i] = list(map(lambda i: dict_keep[i], C.children[i]))
    C.size = T.size[keep].copy()
    for i in T.path_to_root(k)[:-1]:
        C.size[dict_keep[i]] -= to_substrack
    C.depth = T.depth[keep].copy()
    C.weight = T.weight[keep].copy()
    if renormalise_weights:
        C.weight = standardisation(C.weight)
    return C


def tree_add_node(T: RecursiveTree, k: int, **kwargs) -> None:
    """
    Adds a node above k.
    """
    n = T.number_of_vertices
    T.parent[n] = k
    T.depth[n] = T.depth[k] + 1
    T.children[n] = []
    T.children[k] += [n]
    if "map_weights" in kwargs:
        func = kwargs["map_weights"]
        T.weight[:n] = np.vectorize(func)(T.weight[:n])
    for j in T.path_to_root(n):
        T.size[j] += 1
    if "new_weight" in kwargs:
        T.weight[n] = kwargs["new_weight"]
    else:
        T.weight[n] = 1
