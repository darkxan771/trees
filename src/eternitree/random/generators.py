from typing import Callable
from typing import Sequence

import numpy as np
import numpy.random as rand
from scipy.special import factorial

from ..abstraction import IntegerPartition
from ..abstraction import SetPartition
from ..abstraction.helpers import standardisation
from ..containers.generating_series import generating_series_SP
from ..recursive import RecursiveTree


def _random_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    alea_w = (np.arange(1, n) + np.arange(1, n) ** 2) * rand.random(size=n - 1)
    w = np.floor(np.sqrt(alea_w + 0.25) + 0.5)
    J = 1 + rand.randint(w)
    return (w.astype(int), J)


def _RSK_P(w: Sequence[int]) -> list:
    def insert_in_row(x: int, R: list[int]) -> tuple[list[int], None | int]:
        if all(x >= r for r in R):
            return R + [x], None
        else:
            i = R.index([r for r in R if r > x][0])
            return R[:i] + [x] + R[i + 1 :], R[i]

    res = [[w[0]]]
    for x in w[1:]:
        IR = insert_in_row(x, res[0])
        res[0] = IR[0]
        k = 1
        while IR[1] is not None:
            if k + 1 > len(res):
                res.append([IR[1]])
                IR = [], None
            else:
                IR = insert_in_row(IR[1], res[k])
                res[k] = IR[0]
                k += 1
    return res


def set_partition_uniform(L: Sequence[int]) -> SetPartition:
    """
    Generates a random set partition of L with uniform distribution.
    """
    S = list(L)
    S.sort()
    n = len(S)
    bn = generating_series_SP(n)[-1]
    k = 0
    count = 0
    alea = rand.random()
    while alea > count:
        k += 1
        count += k**n / (factorial(k, exact=True) * np.exp(1) * bn)
    C = rand.randint(low=0, high=k, size=n)
    res = []
    for j in range(k):
        toadd = [S[i] for i in range(n) if C[i] == j]
        if len(toadd) > 0:
            res.append(toadd)
    return SetPartition(res)


def set_partition_ewens(L: Sequence[int], theta: float) -> SetPartition:
    """
    Generates a random set partition of L with Ewens distribution with
    parameter theta.
    """
    S = list(L)
    S.sort()
    n = len(S)
    res = SetPartition([[S[0]]])
    for k in range(1, n):
        alea = rand.random()
        if alea <= theta / (theta + k):
            res.add_part(S[k])
        else:
            P = np.array(list(len(p) for p in res.parts)) / k
            i = rand.choice(res.length, p=P)
            res.dict[i].append(S[k])
    return res


def partition_uniform(n: int) -> IntegerPartition:
    """
    Generates a random integer partition with size n and
    uniform distribution.
    """
    from ..boltzmann.boltzmann_partition import boltzmann_sampler
    from ..boltzmann.boltzmann_partition import find_x_for_n

    x = find_x_for_n(n)
    res = boltzmann_sampler(x)
    while res.size != n:
        res = boltzmann_sampler(x)
    return res


def partition_plancherel(n: int) -> IntegerPartition:
    """
    Generates a random integer partition with size n and
    Plancherel distribution.
    """
    U = rand.random(size=n)
    perm = standardisation(U).tolist()
    return IntegerPartition([len(R) for R in _RSK_P(perm)])


def partition_ewens(n: int, theta: float) -> IntegerPartition:
    """
    Generates a random integer partition with size n and
    Ewens distribution with parameter theta.
    """
    res = []
    for k in range(n):
        test = rand.random() <= theta / (theta + k)
        if test:
            res.append(1)
        else:
            ind = rand.choice(len(res), p=np.array(res) / k)
            res[ind] += 1
    res.sort(reverse=True)
    return IntegerPartition(res)


def tree_subtree(U) -> RecursiveTree:
    """
    Picks at random a subtree T of the supertree U.
    """
    T = U.get_random_element()
    k = T.random_node()
    return T.subtree(k, renormalise_weights=T.is_double_recursive())


def tree_cut(U) -> RecursiveTree:
    """
    Picks at random a cut T of the supertree U.
    """
    T = U.get_random_element()
    k = rand.randint(1, U.size)
    return T.cut(
        T.subtree_indices(k), renormalise_weights=T.is_double_recursive()
    )


def recursive_tree_uniform(n: int) -> RecursiveTree:
    """
    Picks at random a recursive tree with size n, under the uniform
    distribution.
    """
    T = RecursiveTree(max_size=n)
    for k in range(1, n):
        T.add_node(rand.randint(0, k))
    return T


def tree_plancherel(n: int) -> RecursiveTree:
    """
    Picks at random a recursive tree with size n, under the Plancherel
    distribution.
    """
    T = RecursiveTree(max_size=n)
    L = [0]
    w, J = _random_pairs(n)
    for k in range(1, n):
        i = L[k - w[k - 1]]
        T.add_node(i)
        L.insert(k + 1 - J[k - 1], k)
    for w in range(1, n + 1):
        T.weight[L[n - w]] = w
    return T


def tree_weighted(
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


def tree_ewens(n: int, theta: float) -> RecursiveTree:
    """
    Picks at random a recursive tree with size n, under the Ewens
    distribution with parameter theta.
    """
    T = RecursiveTree(max_size=n)
    T.weight[:n] = np.ones(n, dtype=int)
    T.n[0] = n
    if n > 1:
        L = partition_ewens(n - 1, theta)
        subtrees = [tree_ewens(k, theta) for k in L.parts]
        perm = rand.permutation(np.arange(1, n))
        c = 0
        for i, k in enumerate(L.parts):
            perm[c : c + k] = np.sort(perm[c : c + k])
            loc_dict = perm[c : c + k]
            loc_tree = subtrees[i]
            T.parent[perm[c]] = 0
            T.parent[perm[c + 1 : c + k]] = loc_dict[loc_tree.parent[1:k]]
            T.n[perm[c : c + k]] = loc_tree.n[:k]
            T.depth[perm[c : c + k]] = loc_tree.depth[:k] + 1
            c += k
    return T


generate_random: dict[tuple[str, str], Callable] = {
    ("set partition", "uniform"): lambda S, _: set_partition_uniform(S),
    ("set partition", "ewens"): set_partition_ewens,
    ("partition", "uniform"): lambda n, _: partition_uniform(n),
    ("partition", "ewens"): partition_ewens,
    ("partition", "plancherel"): lambda n, _: partition_plancherel(n),
    ("recursive tree", "deterministic"): lambda _, T: T,
    ("recursive tree", "cut"): lambda _, U: tree_cut(U),
    ("recursive tree", "subtree"): lambda _, U: tree_subtree(U),
    ("recursive tree", "uniform"): lambda n, _: recursive_tree_uniform(n),
    ("recursive tree", "plancherel"): lambda n, _: tree_plancherel(n),
    ("recursive tree", "weighted"): tree_weighted,
    ("recursive tree", "ewens"): tree_ewens,
}
