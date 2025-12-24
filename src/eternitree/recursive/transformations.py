# Useful functions for the transformation of recursive trees


import numpy as np

from ..abstraction.helpers import shift_array
from ..abstraction.helpers import shift_dict
from ..abstraction.helpers import standardisation
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
    S.size = T.size[sub].copy()
    S.depth = T.depth[sub].copy() - T.depth[k]
    S.weight = T.weight[sub].copy()
    if renormalise_weights:
        S.weight = standardisation(S.weight)
    if k in T.birth_times.keys():
        S.birth_times = {}
        for l in sub:
            S.birth_times[dict_sub[l]] = T.birth_times[l] - T.birth_times[k]
            S.birth_processes[dict_sub[l]] = T.birth_processes[l].copy()
    return S


def tree_cut(
    T: RecursiveTree, L: list[int], renormalise_weights: bool = False
) -> RecursiveTree:
    """
    Removes the nodes in L if the complement is a recursive part of T.
    """
    n = T.number_of_vertices
    keep = [k for k in range(n) if (not (k in L))]
    if not T.has_recursive_part(keep):
        raise ValueError("One cannot cut the specified part")
    dict_keep = {keep[i]: i for i in range(len(keep))}
    dict_keep[-1] = -1
    C = RecursiveTree(max_size=len(keep))
    for k in keep[1:]:
        C.add_node(dict_keep[T.parent[k]], new_weight=T.weight[k])
    if renormalise_weights:
        C.weight = standardisation(C.weight)
    for l in keep:
        if l in T.birth_times.keys():
            C.birth_times[dict_keep[l]] = T.birth_times[l]
            C.birth_processes[dict_keep[l]] = T.birth_processes[l].copy()
    return C


def tree_add_node(T: RecursiveTree, k: int, **kwargs) -> None:
    """
    Adds a node above k.
    """
    n = T.number_of_vertices
    T.parent[n] = k
    T.depth[n] = T.depth[k] + 1
    if "map_weights" in kwargs:
        func = kwargs["map_weights"]
        T.weight[:n] = np.vectorize(func)(T.weight[:n])
    for j in T.path_to_root(n):
        T.size[j] += 1
    if "new_weight" in kwargs:
        T.weight[n] = kwargs["new_weight"]
    else:
        T.weight[n] = 1
    if "birth_time" in kwargs:
        T.birth_times[n] = kwargs["birth_time"]
    if "birth_process" in kwargs:
        T.birth_processes[n] = kwargs["birth_process"]


def tree_insert_node(T: RecursiveTree, l: int, parent: int, **kwargs) -> None:
    """
    Shifts all the indices greater than l by one, and
    inserts then a l-th node above the parent.
    """
    n = T.number_of_vertices
    if l > n or n + 1 > T.limit:
        raise ValueError(
            f"One cannot insert the {l}-th node, the tree is not large enough"
        )
    if l <= parent:
        raise ValueError(f"One cannot make {l} a child of {parent}")
    shift_array(T.parent, l, n)
    for k in range(n + 1):
        T.parent[k] += int(T.parent[k] >= l)
    shift_array(T.size, l, n)
    shift_array(T.depth, l, n)
    shift_array(T.weight, l, n)
    T.birth_times = shift_dict(T.birth_times, l, n)
    T.birth_processes = shift_dict(T.birth_processes, l, n)
    T.parent[l] = parent
    T.weight[l] = 1
    T.depth[l] = T.depth[parent] + 1
    T.size[l] = 0
    for k in T.path_to_root(l):
        T.size[k] += 1
    if "birth_time" in kwargs:
        T.birth_times[l] = kwargs["birth_time"]
    if "birth_process" in kwargs:
        T.birth_processes[l] = kwargs["birth_process"]
