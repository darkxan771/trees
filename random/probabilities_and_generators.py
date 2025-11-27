# Useful functions for the manipulation of random trees

import numpy as np
import numpy.random as rand

from scipy.special import factorial
from typing import Callable
from copy import deepcopy
from .crump_jagers_mode import PointProcess
from .random_partitions import EwensPartition
from ..recursive import RecursiveTree
from ..rooted import RootedTree
from ..containers import RootedTrees, RecursiveTrees


def _random_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    alea_w = (np.arange(1, n) + np.arange(1, n) ** 2) * rand.random(size=n - 1)
    w = np.floor(np.sqrt(alea_w + 0.25) + 0.5)
    J = 1 + rand.randint(w)
    return (w.astype(int), J)


def probability_uniform_rooted(T: RootedTree) -> float:
    """
    Returns the probability of the rooted tree T under the
    uniform distribution on trees with the same size.
    """
    return float(1 / RootedTrees(T.size).cardinality())


def probability_uniform_recursive(T: RecursiveTree) -> float:
    """
    Returns the probability of the recursive tree T under
    the uniform distribution on trees with the same size.
    """
    return float(1 / RecursiveTrees(T.size[0]).cardinality())


def probability_plancherel(T: RecursiveTree) -> float:
    """
    Returns the probability of the recursive tree T under
    the Plancherel distribution.
    """
    n = T.size[0]
    num = int(factorial(n) / np.prod(T.size[:n]))
    denum = np.prod(np.array([(i * (i + 1) / 2) for i in range(1, n)]))
    return float(num / denum)


def probability_weighted(
    T: RecursiveTree, weight_function: Callable[[int], float]
) -> float:
    """
    Returns the probability of the recursive tree T under the
    weighted distribution.
    """
    n = T.size[0]
    W = np.vectorize(weight_function)
    num = np.prod(W(T.parent[np.arange(1, n)]))
    denom = np.prod(np.cumsum(W(np.arange(n - 1))))
    return float(num / denom)


def probability_ewens_tree(T: RecursiveTree, theta: float) -> float:
    """
    Returns the probability of the recursive tree T under
    the Ewens distribution with parameter theta.
    """
    if T.size[0] == 1:
        return float(1)
    else:
        sub = [T.subtree(k) for k in T.children[0]]
        A = float(np.prod([probability_ewens_tree(U, theta) for U in sub]))
        p = EwensPartition(T.size[0] - 1, theta).probability(
            T.subtrees_partition
        )
        return A * p / T.subtrees_partition.bell_number


compute_probabilities: dict[tuple[str, str], Callable] = {
    ("Deterministic", "recursive"): (lambda T, U: float(T == U)),
    ("Uniform", "rooted"): (lambda T, _: probability_uniform_rooted(T)),
    ("Uniform", "recursive"): (lambda T, _: probability_uniform_recursive(T)),
    ("Plancherel", "recursive"): (lambda T, _: probability_plancherel(T)),
    ("Weighted", "recursive"): probability_weighted,
    ("Ewens", "recursive"): probability_ewens_tree,
}


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
    R = RecursiveTree(max_size=n)
    Lpp = [deepcopy(pp)]
    birth = [0]
    Lpp[0].times = []
    next_computed = []
    m = 1
    while m < n:
        for i in range(m):
            if i not in [x[0] for x in next_computed]:
                next_computed.append((i, birth[i] + next(Lpp[i])))
        next_computed.sort(key=lambda x: x[1])
        i, t = next_computed.pop(0)
        R.add_node(i)
        m += 1
        birth.append(t)
        Lpp.append(deepcopy(pp))
        Lpp[-1].times = []
    last = birth[n - 1]
    for i in range(n):
        L = [birth[i] + t for t in Lpp[i].times if birth[i] + t < last]
        R.additional[i] = np.array(L)
    return R


generators_random_tree: dict[tuple[str, str], Callable] = {
    ("Deterministic", "recursive"): (lambda _, T: T),
    ("Subtree", "recursive"): (lambda _, U: random_subtree(U)),
    ("Cut", "recursive"): (lambda _, U: random_cut(U)),
    ("Uniform", "recursive"): (lambda n, _: uniform_recursive_tree(n)),
    ("Plancherel", "recursive"): (lambda n, _: plancherel_tree(n)),
    ("Weighted", "recursive"): weighted_recursive_tree,
    ("Ewens", "recursive"): ewens_recursive_tree,
    ("Crump-Jagers-Mode", "recursive"): CJM_recursive_tree,
}
