import numpy as np
import networkx as nx
from collections.abc import Sequence
from typing import Callable
from .rooted_tree import RootedTree
from ..recursive.recursive_tree import RecursiveTree


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


def tree_to_nested_list(T: RootedTree) -> list:
    """
    Converts the rooted tree to a nested list.
    """
    return [tree_to_nested_list(child) for child in T.children]


def tree_to_code(T: RootedTree) -> tuple:
    """
    Returns the code of the rooted tree (level sequence in the depth
    first search).
    """
    return nested_list_to_code(tree_to_nested_list(T))


def _insert_in_recursive_tree(
    T: RootedTree, RT: RecursiveTree, d: int, mini: int, maxi: int
) -> None:
    sizes = [c.size for c in T.children]
    RT.children[mini] = (
        mini + 1 + np.cumsum(np.array([0] + sizes))[:-1]
    ).tolist()
    RT.weight[mini] = 1
    RT.depth[mini] = d
    RT.size[mini] = maxi - mini
    if len(sizes) > 0:
        mini2 = mini + 1
        for c in T.children:
            maxi2 = mini2 + c.size
            RT.parent[mini2] = mini
            _insert_in_recursive_tree(c, RT, d + 1, mini2, maxi2)
            mini2 = maxi2


def tree_to_recursive(T: RootedTree, random: bool = False) -> RecursiveTree:
    """
    Converts the rooted tree to a recursive tree. The increasing
    labelling which is chosen is uniformly distributed over all
    possibilities.
    """
    RT = RecursiveTree(max_size=T.size)
    _insert_in_recursive_tree(T, RT, 0, 0, T.size)
    if random:
        RT.random_relabelling()
    return RT


def tree_to_networkx(T: RootedTree) -> nx.classes.digraph.DiGraph:
    """
    Encodes the tree in a NetworkX labelled digraph.
    """
    RT = tree_to_recursive(T)
    return RT.convert("networkx")


compute_conversions: dict[str, Callable] = {
    "code": tree_to_code,
    "networkx": tree_to_networkx,
    "rooted": (lambda T: T),
    "recursive": tree_to_recursive,
    "nested_list": tree_to_nested_list,
}
