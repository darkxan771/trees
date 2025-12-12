# Useful functions for the computation of probabilities of random trees

from typing import Callable

import numpy as np
from scipy.special import factorial

from ..abstraction import Tree
from ..containers import RecursiveTrees
from ..containers import RootedTrees
from ..recursive import RecursiveTree
from ..rooted import RootedTree
from .random_partitions import EwensPartition


def probability_uniform_rooted(T: Tree) -> float:
    """
    Returns the probability of the rooted tree T under the
    uniform distribution on trees with the same size.
    """
    return float(1 / RootedTrees(T.number_of_vertices).cardinality)


def probability_uniform_recursive(T: Tree) -> float:
    """
    Returns the probability of the recursive tree T under
    the uniform distribution on trees with the same size.
    """
    return float(1 / RecursiveTrees(T.number_of_vertices).cardinality)


def probability_plancherel(T: Tree) -> float:
    """
    Returns the probability of the recursive or rooted tree T under
    the Plancherel distribution (beware that the result is not the same).
    """
    if isinstance(T, RecursiveTree):
        n = T.number_of_vertices
        num = int(factorial(n) / np.prod(T.size[:n]))
        denum = np.prod(np.array([(i * (i + 1) / 2) for i in range(1, n)]))
        return float(num / denum)
    elif isinstance(T, RootedTree):
        return T.plancherel_measure
    else:
        raise NotImplementedError


def probability_weighted(
    T: Tree, weight_function: Callable[[int], float]
) -> float:
    """
    Returns the probability of the recursive tree T under the
    weighted distribution.
    """
    if isinstance(T, RecursiveTree):
        n = T.number_of_vertices
        W = np.vectorize(weight_function)
        num = np.prod(W(T.parent[np.arange(1, n)]))
        denom = np.prod(np.cumsum(W(np.arange(n - 1))))
        return float(num / denom)
    else:
        raise NotImplementedError


def probability_ewens_tree(T: Tree, theta: float) -> float:
    """
    Returns the probability of the recursive tree T under
    the Ewens distribution with parameter theta.
    """
    if T.number_of_vertices == 1:
        return float(1)
    else:
        A = float(
            np.prod([probability_ewens_tree(U, theta) for U in T.subtree_list])
        )
        p = EwensPartition(T.number_of_edges, theta).probability(
            T.subtrees_partition
        )
        return A * p / T.subtrees_partition.bell_number


compute_probabilities: dict[tuple[str, str], Callable] = {
    ("Deterministic", "recursive"): lambda T, U: float(T == U),
    ("Uniform", "rooted"): lambda T, _: probability_uniform_rooted(T),
    ("Uniform", "recursive"): lambda T, _: probability_uniform_recursive(T),
    ("Plancherel", "recursive"): lambda T, _: probability_plancherel(T),
    ("Weighted", "recursive"): probability_weighted,
    ("Ewens", "recursive"): probability_ewens_tree,
}
