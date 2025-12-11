# Useful functions for the generation of random trees

from typing import Callable

import numpy as np
import numpy.random as rand

from ..recursive import RecursiveTree
from .crump_jagers_mode import PointProcess
from .random_partitions import EwensPartition


def _random_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    alea_w = (np.arange(1, n) + np.arange(1, n) ** 2) * rand.random(size=n - 1)
    w = np.floor(np.sqrt(alea_w + 0.25) + 0.5)
    J = 1 + rand.randint(w)
    return (w.astype(int), J)


def random_subtree(U) -> RecursiveTree:
    """
    Picks at random a subtree T of the supertree U.
    """
    T = U.get_random_element()
    if isinstance(T, RecursiveTree):
        k = T.random_node()
        return T.subtree(k, renormalise_weights=T.is_double_recursive())
    else:
        raise NotImplementedError


def random_cut(U) -> RecursiveTree:
    """
    Picks at random a cut T of the supertree U.
    """
    T = U.get_random_element()
    if isinstance(T, RecursiveTree):
        k = rand.randint(1, U.size)
        return T.cut(k, renormalise_weights=T.is_double_recursive())
    else:
        raise NotImplementedError


def uniform_recursive_tree(n: int) -> RecursiveTree:
    """
    Picks at random a recursive tree with size n, under the uniform
    distribution.
    """
    T = RecursiveTree(max_size=n)
    for k in range(1, n):
        T.add_node(rand.randint(0, k))
    return T


def plancherel_tree(n: int) -> RecursiveTree:
    """
    Picks at random a recursive tree with size n, under the Plancherel
    distribution.
    """
    T = RecursiveTree(max_size=n)
    L = [0]
    (w, J) = _random_pairs(n)
    for k in range(1, n):
        i = L[k - w[k - 1]]
        T.add_node(i)
        L.insert(k + 1 - J[k - 1], k)
    for w in range(1, n + 1):
        T.weight[L[n - w]] = w
    return T


def weighted_recursive_tree(
    n: int, weight_function: Callable[[int], float]
) -> RecursiveTree:
    """
    Picks at random a recursive tree with size n, under the weighted
    distribution with weight function w.
    """

    T = RecursiveTree(max_size=n)
    T.weight[0] = weight_function(0)
    for k in range(1, n):
        i = T.random_node(with_weights=True)
        T.add_node(i, new_weight=weight_function(k))
    return T


def ewens_recursive_tree(n: int, theta: float) -> RecursiveTree:
    """
    Picks at random a recursive tree with size n, under the Ewens
    distribution with parameter theta.
    """
    T = RecursiveTree(max_size=n)
    T.weight[:n] = np.ones(n, dtype=int)
    T.size[0] = n
    if n > 1:
        L = EwensPartition(n - 1, theta).get_random_element()
        subtrees = [ewens_recursive_tree(k, theta) for k in L.parts]
        perm = rand.permutation(np.arange(1, n))
        c = 0
        for i, k in enumerate(L.parts):
            perm[c : c + k] = np.sort(perm[c : c + k])
            loc_dict = perm[c : c + k]
            loc_tree = subtrees[i]
            T.parent[perm[c]] = 0
            T.parent[perm[c + 1 : c + k]] = loc_dict[loc_tree.parent[1:k]]
            T.size[perm[c : c + k]] = loc_tree.size[:k]
            T.depth[perm[c : c + k]] = loc_tree.depth[:k] + 1
            T.children[0].append(perm[c])
            for j, x in enumerate(perm[c : c + k]):
                T.children[x] = (loc_dict[loc_tree.children[j]]).tolist()
            c += k
    return T


def CJM_recursive_tree(n: int, pp: PointProcess):
    """
    Picks at random a recursive tree with size n, chosen according to a
    Crump-Jagers-Mode branching process.
    """
    base = pp
    R = RecursiveTree()
    return R.resize(n)


compute_random_trees: dict[tuple[str, str], Callable] = {
    ("Deterministic", "recursive"): lambda _, T: T,
    ("Subtree", "recursive"): lambda _, U: random_subtree(U),
    ("Cut", "recursive"): lambda _, U: random_cut(U),
    ("Uniform", "recursive"): lambda n, _: uniform_recursive_tree(n),
    ("Plancherel", "recursive"): lambda n, _: plancherel_tree(n),
    ("Weighted", "recursive"): weighted_recursive_tree,
    ("Ewens", "recursive"): ewens_recursive_tree,
    ("Crump-Jagers-Mode", "recursive"): CJM_recursive_tree,
}
