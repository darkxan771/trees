import numpy as np
import numpy.random as rand
import scipy.stats as scs

from .recursive_trees import RecursiveTree
from .rooted_trees import RootedTree
from .routines import random_pairs, poisson_galton_watson
from .boltzmann import compute_values, find_x_for_n

from scipy.special import factorial


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

    def container(self):
        """
        The support of the distribution of random trees.
        """
        from .containers import RecursiveTrees_n, RootedTrees_n

        if self.type == "recursive":
            return RecursiveTrees_n(self.size)
        else:
            return RootedTrees_n(self.size)

    def probability(self, T) -> float:
        """
        Returns the probability of the tree T under
        the distribution considered.
        """
        if T in self.container():
            return float(1 / self.container().cardinality())
        else:
            return float(0)


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
        N = [0] + [scs.poisson(values[k * i] / k).rvs() for k in div_range]
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
                        scs.poisson(values[k] / k).rvs() for k in range(2, 51)
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

    def probability(self, T) -> float:
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
        (w, J) = random_pairs(n)
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

    def __init__(self, n, w):
        self.size = n
        self.type = "recursive"
        self.weight = w

    def __repr__(self):
        return f"Weighted Recursive Tree with size {self.size}"

    def probability(self, T) -> float:
        """
        Returns the probability of the recursive tree T under the
        weighted distribution.
        """
        if T in self.container():
            n = self.size
            num = np.prod(T.weight[T.parent[np.arange(1, n)]])
            denom = np.prod(np.cumsum([T.weight[: n - 1]]))
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
            T.add_node(i, new_weight=self.weight[k])
        return T
