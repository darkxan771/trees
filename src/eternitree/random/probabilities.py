# Useful functions for the computation of probabilities of random objects

from typing import Callable

import numpy as np
from scipy.special import factorial

from ..abstraction import IntegerPartition
from ..abstraction import SetPartition
from ..containers import IntegerPartitions
from ..containers import RecursiveTrees
from ..containers import RootedTrees
from ..containers.set_partitions import bell_number
from ..recursive import RecursiveTree


def probability_ewens(L: IntegerPartition, theta: float) -> float:
    """
    Returns the probability of L under the Ewens measure with
    parameter theta.
    """
    n = L.size
    A = np.prod((1 + np.arange(n)) / (theta + np.arange(n)))
    B = theta**L.length / L.z
    return float(A * B)


def probability_sp_ewens(P: SetPartition, theta: float) -> float:
    """
    Returns the probability of P under the Ewens measure with
    parameter theta.
    """
    n = P.size
    m = P.type.dictionary
    res = theta**P.length / np.prod(theta + np.arange(n))
    for i in m.keys():
        res *= factorial(i - 1, exact=True) ** m[i]
    return float(res)


def probability_tree_plancherel(T: RecursiveTree) -> float:
    """
    Returns the probability of the recursive or rooted tree T under
    the Plancherel distribution (beware that the result is not the same).
    """
    n = T.number_of_vertices
    num = int(factorial(n) / np.prod(T.size[:n]))
    denum = np.prod(np.array([(i * (i + 1) / 2) for i in range(1, n)]))
    return float(num / denum)


def probability_tree_weighted(
    T: RecursiveTree, weight_function: Callable[[int], float]
) -> float:
    """
    Returns the probability of the recursive tree T under the
    weighted distribution.
    """
    n = T.number_of_vertices
    W = np.vectorize(weight_function)
    num = np.prod(W(T.parent[np.arange(1, n)]))
    denom = np.prod(np.cumsum(W(np.arange(n - 1))))
    return float(num / denom)


def probability_tree_ewens(T: RecursiveTree, theta: float) -> float:
    """
    Returns the probability of the recursive tree T under
    the Ewens distribution with parameter theta.
    """
    if T.number_of_vertices == 1:
        return float(1)
    else:
        A = float(
            np.prod([probability_tree_ewens(U, theta) for U in T.subtree_list])
        )
        p = probability_ewens(T.subtrees_partition, theta)
        return A * p / T.subtrees_partition.bell_number


compute_probabilities: dict[tuple[str, str], Callable] = {
    ("set partition", "uniform"): lambda P, _: float(1 / bell_number(P.size)),
    ("set partition", "ewens"): probability_sp_ewens,
    ("partition", "uniform"): lambda L, _: float(
        1 / IntegerPartitions(L.size).cardinality
    ),
    ("partition", "ewens"): probability_ewens,
    ("partition", "plancherel"): lambda L, _: float(
        L.dimension**2 / factorial(L.size, exact=True)
    ),
    ("rooted tree", "uniform"): lambda T, _: float(
        1 / RootedTrees(T.number_of_vertices).cardinality
    ),
    ("recursive tree", "deterministic"): lambda T, U: float(T == U),
    ("recursive tree", "uniform"): lambda T, _: float(
        1 / RecursiveTrees(T.number_of_vertices).cardinality
    ),
    ("recursive tree", "plancherel"): lambda T, _: probability_tree_plancherel(
        T
    ),
    ("recursive tree", "weighted"): probability_tree_weighted,
    ("recursive tree", "ewens"): probability_tree_ewens,
}
