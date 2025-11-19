from __future__ import annotations
from collections import defaultdict
from typing import Callable, TYPE_CHECKING
from scipy.stats import poisson
from scipy.special import factorial
import numpy as np
import numpy.random as rand

from .recursive_trees import RecursiveTree
from .rooted_trees import RootedTree
from .boltzmann import compute_values, find_x_for_n


if TYPE_CHECKING:
    from .containers import RootedTrees, RecursiveTrees


def _random_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    alea_w = (np.arange(1, n) + np.arange(1, n) ** 2) * rand.random(size=n - 1)
    w = np.floor(np.sqrt(alea_w + 0.25) + 0.5)
    J = 1 + rand.randint(w)
    return (w.astype(int), J)


def poisson_galton_watson(mu: float) -> tuple[int, list]:
    """
    Returns the data required to create a Galton-Watson tree
    with offspring distribution Poisson(mu).
    """
    xi = poisson(mu).rvs()
    subs = [poisson_galton_watson(mu) for _ in range(xi)]
    size = 1 + sum([c[0] for c in subs])
    return size, subs


class RandomTree:
    """
    A generic class for distributions of random trees (either rooted
    unlabelled trees, or rooted recursive trees).
    """

    def __init__(self, n: int, type: str = "recursive"):
        self.size = n
        self.type = type

    def __repr__(self):
        return f"Random {self.type} tree with size {self.size}"

    def container(self) -> RootedTrees | RecursiveTrees:
        """
        The support of the distribution of random trees.
        """
        from .containers import RootedTrees, RecursiveTrees

        if self.type == "recursive":
            return RecursiveTrees(self.size)
        else:
            return RootedTrees(self.size)

    def probability(self, T) -> float:
        """
        Returns the probability of the tree T under
        the distribution considered.
        """
        if T in self.container():
            return float(1 / self.container().cardinality())
        else:
            return float(0)

    def distribution(self) -> defaultdict:
        """
        Returns a dictionary with items (T, probability[T]), where T is
        identified by its code.
        """
        if self.size == 0:
            raise NotImplementedError
        return defaultdict(
            float, {T.to_code(): self.probability(T) for T in self.container()}
        )

    def distribution_partition(self) -> defaultdict:
        """
        Returns a dictionary with items (L, probability[L]), where L
        runs over the set of integer partitions with size n - 1, and
        probability[L] is the probability of L being the list of
        sizes of the subtrees of a random tree.
        """
        if self.size == 0:
            raise NotImplementedError
        res = defaultdict(float)
        for T in self.container():
            p = tuple(T.subtrees_partition.parts)
            res[p] += self.probability(T)
        return res

    def get_random_element(self) -> RootedTree | RecursiveTree:
        raise NotImplementedError


class DeterministicRecursiveTree(RandomTree):
    """
    A deterministic recursive tree.
    """

    def __init__(self, T: RecursiveTree):
        self.size = T.size[0]
        self.type = "recursive"
        self.tree = T

    def __repr__(self):
        return self.tree.__repr__()

    def probability(self, T: RecursiveTree) -> float:
        """
        Returns 1 if T is the deterministic recursive tree,
        and 0 otherwise.
        """
        return float(self.tree == T)

    def get_random_element(self) -> RecursiveTree:
        """
        Returns the deterministic recursive tree.
        """
        return self.tree


class RandomSubtree(RandomTree):
    """
    Random subtree T of a supertree U, which can itself be random (but with
    fixed size).
    """

    def __init__(self, U: RandomTree):
        self.size = 0
        self.type = "recursive"
        self.supertree = U
        if not U.type == "recursive":
            raise ValueError("U is not a random tree with recursive type")

    def __repr__(self):
        return f"Random subtree of a {self.supertree}"

    def distribution(self) -> defaultdict:
        """
        Returns a dictionary with items (T, probability[T]), where T is
        identified by its code.
        """
        from .containers import RecursiveTrees

        d = defaultdict(float)
        n = self.supertree.size
        for k in range(n):
            for T in RecursiveTrees(n):
                d[T.subtree(k).to_code()] += self.supertree.probability(T) / n
        return d

    def probability(self, T: RecursiveTree) -> float:
        """
        Returns the probability of the recursive tree T as a
        random subtree of the supertree U.
        """
        return self.distribution()[T.to_code()]

    def get_random_element(self) -> RecursiveTree:
        """
        Picks at random a subtree T of the supertree U.
        """
        T = self.supertree.get_random_element()
        if isinstance(T, RecursiveTree):
            k = T.random_node()
            renorm = False
            if T.is_double_recursive():
                renorm = True
            return T.subtree(k, renormalise_weights=renorm)
        else:
            raise NotImplementedError


class RandomCut(RandomTree):
    """
    Random cut T of a supertree U, which can itself be random.
    """

    def __init__(self, U: RandomTree):
        self.size = 0
        self.type = "recursive"
        self.supertree = U
        if not U.type == "recursive":
            raise ValueError("T is not a random tree with recursive type")

    def __repr__(self):
        return f"Random cut of a {self.supertree}"

    def distribution(self) -> defaultdict:
        from .containers import RecursiveTrees

        d = defaultdict(float)
        n = self.supertree.size
        for k in range(1, n):
            for T in RecursiveTrees(n):
                code = T.cut(k).to_code()
                d[code] += self.supertree.probability(T) / (n - 1)
        return d

    def probability(self, T: RecursiveTree) -> float:
        """
        Returns the probability of the recursive tree T as a
        random cut of the supertree U.
        """
        return self.distribution()[T.to_code()]

    def get_random_element(self) -> RecursiveTree:
        """
        Picks at random a cut T of the supertree U.
        """
        T = self.supertree.get_random_element()
        if isinstance(T, RecursiveTree):
            k = rand.randint(1, self.supertree.size)
            renorm = False
            if T.is_double_recursive():
                renorm = True
            return T.cut(k, renormalise_weights=renorm)
        else:
            raise NotImplementedError


class UniformRootedTree(RandomTree):
    """
    Class of uniformly distributed random rooted trees with n nodes.
    """

    def __init__(self, n: int):
        self.size = n
        self.type = "rooted"

    def __repr__(self):
        return f"Uniform Rooted Tree with size {self.size}"

    def _sampler_with_precomputed(
        self, values, i: int, pointed: bool = False
    ) -> RootedTree:
        div_range = range(1, 50 // i + 1)
        N = [0] + [poisson(values[k * i] / k).rvs() for k in div_range]
        res = []
        if i == 1 and pointed:
            P = np.array(values)
            P = P / np.sum(P)
            K = rand.choice(len(P), p=P)
            res += [self._sampler_with_precomputed(values, K)] * K
        for k in div_range:
            for _ in range(N[k]):
                res += [self._sampler_with_precomputed(values, k * i)] * k
        return RootedTree(res)

    def _reconstruct(self, s: int, GW: list, other_trees: list) -> RootedTree:
        trees = other_trees[0]
        S = [c[0] for c in GW]
        cuts = np.cumsum(np.array([1] + S))
        L = len(GW)
        CT = [other_trees[cuts[i] : cuts[i + 1]] for i in range(L)]
        trees += [self._reconstruct(S[i], GW[i][1], CT[i]) for i in range(L)]
        return RootedTree(trees)

    def boltzmann_sampler(self, x: float, pointed: bool = False) -> RootedTree:
        """
        Picks at random a rooted unlabelled tree.

        Each tree T has probability x^{|T|} / T(x), where T(x) is the
        generating series of the species of rooted unlabelled trees.

        If the argument pointed is set to True, the random tree has now
        probability |T| x^{|T|} / T.(x), where T.(x) = x T'(x) is the
        generating series of the species of cycle-pointed rooted unlabelled
        trees. This reduces the variance of the size of T.
        """
        values = compute_values(x)
        return self._sampler_with_precomputed(values, 1, pointed)

    def get_random_element(self, exact: bool = True) -> RootedTree:
        """
        Picks at random a rooted unlabelled tree with size n.

        If the parameter exact is set to False, the size n is replaced by a
        random size N in [0.9n, 1.1n]. Conditionnally to N, the distribution
        of the tree is uniform over the set of rooted unlabelled trees with
        size N.
        """
        n = self.size
        if n == 1:
            return RootedTree([])
        elif n == 2:
            return RootedTree([RootedTree([])])
        else:
            x = find_x_for_n(n, True)
            values = compute_values(x)
            test = True
            s, GW = 1, []
            other_trees = []
            while test:
                s, GW = poisson_galton_watson(values[1])
                P = np.array(values)
                P = P / np.sum(P)
                K = rand.choice(len(P), p=P)
                if K == 1:
                    t, GW2 = poisson_galton_watson(values[1])
                    s = s + t
                    GW.append((t, GW2))
                other_trees = []
                S = s
                for _ in range(s):
                    N = [0, 0] + [
                        poisson(values[k] / k).rvs() for k in range(2, 51)
                    ]
                    toadd = []
                    for k in range(2, 51):
                        for _ in range(N[k]):
                            T = self._sampler_with_precomputed(values, k)
                            toadd += [T] * k
                    other_trees.append(toadd)
                    S += sum([T.size for T in toadd])
                if K > 1:
                    T = self._sampler_with_precomputed(values, K)
                    S += T.size * K
                    other_trees[0] += [T] * K
                if ((not exact) and 0.9 < S / n < 1.1) or (S == n):
                    test = False
            return self._reconstruct(s, GW, other_trees)


class PlancherelRecursiveTree(RandomTree):
    """
    Class of Plancherel-distributed random recursive trees with n nodes.
    """

    def __repr__(self):
        return f"Plancherel Recursive Tree with size {self.size}"

    def probability(self, T: RecursiveTree) -> float:
        """
        Returns the probability of the recursive tree T under
        the Plancherel distribution.
        """
        if T in self.container():
            num = int(factorial(self.size) / np.prod(T.size[: self.size]))
            denum = np.prod(
                np.array([(i * (i + 1) / 2) for i in range(1, self.size)])
            )
            return float(num / denum)
        else:
            return float(0)

    def get_random_element(self) -> RecursiveTree:
        """
        Picks at random a recursive tree with size n, under the Plancherel
        distribution.
        """
        n = self.size
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


class UniformRecursiveTree(RandomTree):
    """
    Class of uniformly distributed random recursive trees with n nodes.
    """

    def __repr__(self):
        return f"Uniform Recursive Tree with size {self.size}"

    def get_random_element(self) -> RecursiveTree:
        """
        Picks at random a recursive tree with size n, under the uniform
        distribution.
        """
        n = self.size
        T = RecursiveTree(max_size=n)
        for k in range(1, n):
            T.add_node(rand.randint(0, k))
        return T


class WeightedRecursiveTree(RandomTree):
    """
    Class of random recursive trees with n nodes, chosen according to
    weights given by a function i -> w(i).
    """

    def __init__(self, n: int, weight: Callable[[int], float]):
        self.size = n
        self.type = "recursive"
        self.weight = weight

    def __repr__(self):
        return f"Weighted Recursive Tree with size {self.size}"

    def probability(self, T: RecursiveTree) -> float:
        """
        Returns the probability of the recursive tree T under the
        weighted distribution.
        """
        if T in self.container():
            n = self.size
            W = np.vectorize(lambda x: self.weight(x))
            num = np.prod(W(T.parent[np.arange(1, n)]))
            denom = np.prod(np.cumsum(W(np.arange(n - 1))))
            return float(num / denom)
        else:
            return float(0)

    def get_random_element(self) -> RecursiveTree:
        """
        Picks at random a recursive tree with size n, under the weighted
        distribution.
        """
        n = self.size
        T = RecursiveTree(max_size=n)
        T.weight[0] = self.weight(0)
        for k in range(1, n):
            i = T.random_node(with_weights=True)
            T.add_node(i, new_weight=self.weight(k))
        return T
