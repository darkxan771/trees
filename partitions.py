from __future__ import annotations
from typing import Sequence, Callable
from pandas._libs import iNaT
from scipy.special import factorial
from collections import defaultdict
import numpy as np
import numpy.random as rand

from .conversions import standardisation


def generating_series_P(N: int) -> list[int]:
    """
    Computes the N first terms of the generating series of
    integer partitions.
    """
    divs = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            divs[j].append(i)
    res = [0] * (N + 1)
    sigma = [sum(x, 0) for x in divs]
    res[0] = 1
    for n in range(1, N + 1):
        res[n] = sum([sigma[n - k] * res[k] for k in range(n)]) // n
    return res


class IntegerPartition:
    """
    A class for the manipulation of integer partitions.
    """

    def __init__(self, L: Sequence[int]):
        if any(x <= 0 for x in L):
            raise ValueError(
                "The elements of L should be non-negative integers."
            )
        if any(x < y for x, y in zip(L[:-1], L[1:])):
            raise ValueError(
                "The elements of L should be in non-increasing order."
            )
        self.parts = list(int(x) for x in L)
        self.parts

    def __repr__(self):
        return self.parts.__repr__()

    def __hash__(self):
        return hash(tuple(self.parts))

    def __eq__(self, other):
        A = isinstance(other, IntegerPartition)
        return A and self.parts == other.parts

    @property
    def size(self) -> int:
        """
        The size of the integer partition (sum of its parts).
        """
        return sum(self.parts)

    @property
    def length(self) -> int:
        """
        The length of the integer partition (number of parts).
        """
        return len(self.parts)

    @property
    def dictionary(self) -> dict[int, int]:
        """
        The multiplicities of the integers as parts of the integer partition.
        """
        m = max(self.parts)
        return {i: self.parts.count(i) for i in range(1, m + 1)}

    @property
    def bell_number(self) -> int:
        """
        The number of set partitions of [1,n] with type given by the
        integer partition.
        """
        res = factorial(self.size)
        for i, m in self.dictionary.items():
            res /= factorial(m, exact=True) * (factorial(i, exact=True) ** m)
        return int(res)

    @property
    def z(self) -> int:
        """
        The inverse proportion of permutations of [1,n] with cycle type
        given by the integer partition.
        """
        res = 1
        for i, m in self.dictionary.items():
            res *= factorial(m, exact=True) * (i**m)
        return int(res)

    @property
    def conjugate(self) -> IntegerPartition:
        """
        The conjugate integer partition.
        """
        res = [0] * self.parts[0]
        for i in range(self.parts[0]):
            res[i] = int(np.count_nonzero(np.array(self.parts) > i))
        return IntegerPartition(res)

    @property
    def dimension(self) -> int:
        """
        The number of standard tableaux with shape given by the
        integer partition.
        """
        C = self.conjugate.parts
        L = self.parts
        res = int(factorial(self.size, exact=True))
        for j in range(len(L)):
            for i in range(L[j]):
                res //= L[j] - i + C[i] - j - 1
        return res


def probability_plancherel(L: IntegerPartition) -> float:
    """
    Returns the probability of L under the Plancherel measure.
    """
    return float(L.dimension**2 / factorial(L.size, exact=True))


def probability_ewens(L: IntegerPartition, theta: float) -> float:
    """
    Returns the probability of L under the Ewens measure with
    parameter theta.
    """
    n = L.size
    A = np.prod((1 + np.arange(n)) / (theta + np.arange(n)))
    B = theta**L.length / L.z
    return float(A * B)


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


def generate_plancherel(n: int) -> IntegerPartition:
    """
    Generates a random integer partition with size n and
    Plancherel distribution.
    """
    U = rand.random(size=n)
    perm = standardisation(U).tolist()
    return IntegerPartition([len(R) for R in _RSK_P(perm)])


def generate_ewens(n: int, theta: float) -> IntegerPartition:
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


compute_probabilities: dict[str, Callable] = {
    "plancherel": (lambda L, N: probability_plancherel(L)),
    "ewens": probability_ewens,
}


generate_random: dict[str, Callable] = {
    "plancherel": (lambda n, N: generate_plancherel(n)),
    "ewens": generate_ewens,
}


class _IntegerPartitionsIterator_n:
    def __init__(self, n: int):
        self.size = n
        self.a = [0] * (n + 1)
        self.k = 1
        self.x = 1
        self.y = n - 1
        self.end_of_cycle = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.k == 0 and self.end_of_cycle:
            raise StopIteration
        else:
            if self.end_of_cycle:
                self.end_of_cycle = False
                self.x = self.a[self.k - 1] + 1
                self.k -= 1
            while 2 * self.x <= self.y:
                self.a[self.k] = self.x
                self.y -= self.x
                self.k += 1
            if self.x <= self.y:
                self.a[self.k] = self.x
                self.a[self.k + 1] = self.y
                res = IntegerPartition(self.a[self.k + 1 :: -1])
                self.x += 1
                self.y -= 1
                return res
            else:
                self.a[self.k] = self.x + self.y
                self.y = self.x + self.y - 1
                res = IntegerPartition(self.a[self.k :: -1])
                self.end_of_cycle = True
                return res


class IntegerPartitions:
    """
    A container for integer partitions with a given size n.
    """

    def __init__(self, n: int):
        self.size = n

    def __repr__(self):
        return f"Integer partitions with size {self.size}"

    def __iter__(self):
        return _IntegerPartitionsIterator_n(self.size)

    def __contains__(self, L):
        A = isinstance(L, IntegerPartition)
        return A and L.size == self.size

    def cardinality(self) -> int:
        """
        Returns the cardinality of the set of integer partitions with
        size n.

        The cardinality satisfies the recurrence relation:

        P[n]
        = 1/n sum([d * P[n-k] for k in 1..n and for d | k]).
        """
        return int(generating_series_P(self.size)[-1])


class RandomPartition:
    def __init__(self, n: int, name: str, parameter: float | None = None):
        self.size = n
        self.name = name
        self.parameter = parameter

    def __repr__(self):
        res = f"Random partition with size {self.size}"
        res += f" and {self.name.capitalize()} distribution"
        if self.parameter is not None:
            res += f"(parameter = {self.parameter})"
        return res

    def probability(self, L: IntegerPartition) -> float:
        if L.size == self.size:
            return compute_probabilities[self.name](L, self.parameter)
        else:
            return float(0)

    def distribution(self) -> defaultdict:
        return defaultdict(
            float,
            {
                tuple(L.parts): self.probability(L)
                for L in IntegerPartitions(self.size)
            },
        )

    def get_random_element(self) -> IntegerPartition:
        return generate_random[self.name](self.size, self.parameter)


def PlancherelPartition(n: int):
    return RandomPartition(n, "plancherel")


def EwensPartition(n: int, theta: float = 1):
    return RandomPartition(n, "ewens", theta)
