# Functions used in the convert methods.

from collections.abc import Sequence
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd

from ..recursive import RecursiveTree
from ..rooted import RootedTree


def recursive_to_code(T: RecursiveTree) -> tuple:
    """
    Returns the code of the recursive tree (list of the nodes of
    attachment).
    """
    n = T.number_of_vertices
    return tuple(T.parent[1:n].tolist())


def code_to_permutation(c: np.ndarray) -> np.ndarray:
    """
    Converts the code array of a recursive tree to a permutation array.
    """
    res = np.zeros(c.size, dtype=int)
    for i in range(1, len(c)):
        res[i] = c[i]
        res[:i] += res[:i] >= res[i]
    return res


def recursive_to_permutation(T: RecursiveTree) -> tuple:
    """
        Returns the permutation of the recursive tree, obtained via one of
    the bijection from recursive trees with size n to permutations with
        size n-1.
    """
    n = T.number_of_vertices
    return tuple((code_to_permutation(T.parent[1:n])).tolist())


def permutation_to_code(p: np.ndarray) -> np.ndarray:
    """
    Converts a permutation array to the code array of a recursive tree.
    """
    res = np.zeros(p.size, dtype=int)
    for i in range(1, len(p)):
        res[i] = np.count_nonzero(p[:i] < p[i])
    return res


def recursive_to_networkx(T: RecursiveTree) -> nx.classes.digraph.DiGraph:
    """
    Encodes the tree in a NetworkX labelled digraph.
    """
    n = T.number_of_vertices
    C = T.children
    G = nx.DiGraph({i: C[i] for i in range(n)})
    for i in range(n):
        G.nodes[i]["label"] = i
        G.nodes[i]["depth"] = T.depth[i]
        G.nodes[i]["weight"] = T.weight[i]
    return G


def recursive_to_rooted(T: RecursiveTree) -> RootedTree:
    """
    Forgets the labels and weights and returns the rooted tree.
    """
    return RootedTree([recursive_to_rooted(U) for U in T.subtree_list])


def recursive_to_dataframe(T: RecursiveTree) -> pd.DataFrame:
    """
    Returns the array encoding the recursive tree, as a Pandas dataframe.
    """
    n = T.number_of_vertices
    matrix = np.array(
        [
            T.parent[:n],
            T.weight[:n],
            T.size[:n],
            T.depth[:n],
            np.array(T.children, dtype=object),
        ]
    )
    row_names = np.array(["parent", "weight", "size", "depth", "children"])
    column_names = np.arange(n)
    return pd.DataFrame(matrix, columns=column_names, index=row_names)


def recursive_to_KP_insertion_array(T: RecursiveTree) -> np.ndarray:
    """
    If the recursive tree is double recursive, returns the
    corresponding Kuba-Panholzer insertion array.
    """
    if not T.is_double_recursive():
        raise ValueError("The tree is not double recursive.")
    else:
        n = T.number_of_vertices
        res = np.zeros((2, n - 1), dtype=int)
        W = T.weights.copy()
        for i in range(1, n):
            J = W[n - i]
            res[:, n - i - 1] = np.array([W[T.parent[n - i]] - 1, J])
            W[: n - i] -= W[: n - i] >= J
        return res


conversions: dict[tuple[str, str], Callable] = {}
conversions[("recursive", "code")] = recursive_to_code
conversions[("recursive", "permutation")] = recursive_to_permutation
conversions[("recursive", "networkx")] = recursive_to_networkx
conversions[("recursive", "rooted")] = recursive_to_rooted
conversions[("recursive", "recursive")] = lambda T: T
conversions[("recursive", "dataframe")] = recursive_to_dataframe
conversions[("recursive", "KP_insertion_array")] = (
    recursive_to_KP_insertion_array
)


def _c_flatten(data):
    for x in data:
        yield from x


def nested_list_to_code(L: list) -> tuple:
    """
    Converts a nested list into the code of the corresponding rooted tree.

    The result is standardised.
    """
    C = [tuple(x + 1 for x in nested_list_to_code(c)) for c in L]
    C.sort(reverse=True)
    return tuple(_c_flatten([(0,)] + C))


def code_to_nested_list(L: Sequence[int]) -> list:
    """
    Converts the code of a rooted tree into the corresponding nested list.

    The result is standardised.
    """
    children = []
    if not (L[0] == 0):
        raise ValueError
    if len(L) > 1:
        child = [0]
        for i in range(2, len(L)):
            if L[i] == 1:
                children.append(child)
                child = [0]
            else:
                child.append(L[i] - 1)
        children.append(child)
    children.sort(key=(lambda x: tuple(x)), reverse=True)
    return [code_to_nested_list(child) for child in children]


def rooted_to_nested_list(T: RootedTree) -> list:
    """
    Converts the rooted tree to a nested list.
    """
    return [rooted_to_nested_list(child) for child in T.subtrees]


def rooted_to_code(T: RootedTree) -> tuple:
    """
    Returns the code of the rooted tree (level sequence in the depth
    first search).
    """
    return nested_list_to_code(rooted_to_nested_list(T))


def _insert_rooted_in_recursive_tree(
    T: RootedTree, RT: RecursiveTree, d: int, mini: int, maxi: int
) -> None:
    sizes = [c.size for c in T.subtrees]
    RT.weight[mini] = 1
    RT.depth[mini] = d
    RT.size[mini] = maxi - mini
    if len(sizes) > 0:
        mini2 = mini + 1
        for c in T.subtrees:
            maxi2 = mini2 + c.size
            RT.parent[mini2] = mini
            _insert_rooted_in_recursive_tree(c, RT, d + 1, mini2, maxi2)
            mini2 = maxi2


def rooted_to_recursive(T: RootedTree, random: bool = False) -> RecursiveTree:
    """
    Converts the rooted tree to a recursive tree. The increasing
    labelling which is chosen is uniformly distributed over all
    possibilities.
    """
    RT = RecursiveTree(max_size=T.size)
    _insert_rooted_in_recursive_tree(T, RT, 0, 0, T.size)
    if random:
        RT.random_relabelling()
    return RT


def rooted_to_networkx(T: RootedTree) -> nx.classes.digraph.DiGraph:
    """
    Encodes the tree in a NetworkX labelled digraph.
    """
    RT = rooted_to_recursive(T)
    return RT.convert("networkx")


conversions[("rooted", "code")] = rooted_to_code
conversions[("rooted", "rooted")] = lambda T: T
conversions[("rooted", "recursive")] = rooted_to_recursive
conversions[("rooted", "networkx")] = rooted_to_networkx
conversions[("rooted", "nested_list")] = rooted_to_nested_list

conversions[("partition", "code")] = lambda P: tuple(P.parts)
conversions[("partition", "dict")] = lambda P: P.dictionary
conversions[("set_partition", "code")] = lambda P: tuple(
    tuple(p) for p in P.parts
)
conversions[("set_partition", "dict")] = lambda P: P.dict
