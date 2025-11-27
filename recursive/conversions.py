# Converts a recursive tree to another format

import numpy as np
import networkx as nx
import pandas as pd
from typing import Callable

from .recursive_tree import RecursiveTree
from ..rooted import RootedTree


def tree_to_code(T: RecursiveTree) -> tuple:
    """
    Returns the code of the recursive tree (list of the nodes of
    attachment).
    """
    return tuple(T.parent[1:].tolist())


def code_to_permutation(c: np.ndarray) -> np.ndarray:
    """
    Converts the code array of a recursive tree to a permutation array.
    """
    res = np.zeros(c.size, dtype=int)
    for i in range(1, len(c)):
        res[i] = c[i]
        res[:i] += res[:i] >= res[i]
    return res


def tree_to_permutation(T: RecursiveTree) -> tuple:
    """
    Returns the permutation of the recursive tree, obtained via one of
    the bijection from recursive trees with size n to permutations with
    size n-1.
    """
    return tuple((code_to_permutation(T.parent[1:])).tolist())


def permutation_to_code(p: np.ndarray) -> np.ndarray:
    """
    Converts a permutation array to the code array of a recursive tree.
    """
    res = np.zeros(p.size, dtype=int)
    for i in range(1, len(p)):
        res[i] = np.count_nonzero(p[:i] < p[i])
    return res


def tree_to_networkx(T: RecursiveTree) -> nx.classes.digraph.DiGraph:
    """
    Encodes the tree in a NetworkX labelled digraph.
    """
    G = nx.DiGraph({i: T.children[i] for i in range(T.size[0])})
    for i in range(T.size[0]):
        G.nodes[i]["label"] = i
        G.nodes[i]["depth"] = T.depth[i]
        G.nodes[i]["weight"] = T.weight[i]
    return G


def tree_to_rooted(T: RecursiveTree) -> RootedTree:
    """
    Forgets the labels and weights and returns the rooted tree.
    """
    subs = T.children[0]
    return RootedTree([tree_to_rooted(T.subtree(c)) for c in subs])


def tree_to_dataframe(T: RecursiveTree) -> pd.DataFrame:
    """
    Returns the array encoding the recursive tree, as a Pandas dataframe.
    """
    D = T.number_of_vertices
    matrix = np.array(
        [
            T.parent[:D],
            T.weight[:D],
            T.size[:D],
            T.depth[:D],
            T.children[:D],
        ]
    )
    row_names = np.array(["parent", "weight", "size", "depth", "children"])
    column_names = np.arange(D)
    return pd.DataFrame(matrix, columns=column_names, index=row_names)


def tree_to_KP_insertion_array(T: RecursiveTree) -> np.ndarray:
    """
    If the recursive tree is double recursive, returns the
    corresponding Kuba-Panholzer insertion array.
    """
    if not T.is_double_recursive():
        raise ValueError("The tree is not double recursive.")
    else:
        n = T.size[0]
        res = np.zeros((2, n - 1), dtype=int)
        W = T.weights.copy()
        for i in range(1, n):
            J = W[n - i]
            res[:, n - i - 1] = np.array([W[T.parent[n - i]] - 1, J])
            W[: n - i] -= W[: n - i] >= J
        return res


compute_conversions: dict[str, Callable] = {
    "code": tree_to_code,
    "permutation": tree_to_permutation,
    "networkx": tree_to_networkx,
    "rooted": tree_to_rooted,
    "recursive": (lambda T: T),
    "dataframe": tree_to_dataframe,
    "KP_insertion_array": tree_to_KP_insertion_array,
}
