# Useful functions for the generation of random partitions

from typing import Callable
from typing import Sequence

import numpy as np
import numpy.random as rand

from ..abstraction import IntegerPartition
from ..abstraction.helpers import standardisation
from ..boltzmann.boltzmann_partition import boltzmann_sampler
from ..boltzmann.boltzmann_partition import find_x_for_n


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


def generate_uniform(n: int) -> IntegerPartition:
    """
    Generates a random integer partition with size n and
    uniform distribution.
    """
    x = find_x_for_n(n)
    res = boltzmann_sampler(x)
    while res.size != n:
        res = boltzmann_sampler(x)
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


generate_random: dict[str, Callable] = {
    "plancherel": lambda n, _: generate_plancherel(n),
    "ewens": generate_ewens,
    "uniform": lambda n, _: generate_uniform(n),
}
