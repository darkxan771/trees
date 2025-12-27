# Constructs a Boltzmann sampler for uniform rooted trees

import numpy as np
import numpy.random as rand
from scipy.optimize import brentq
from scipy.stats import poisson

from ..containers.generating_series import generating_series_T

T353 = generating_series_T(353)
T = T353[:350]
Tprime = [i * T353[i] for i in range(1, 351)]
Tsecond = [(i - 1) * i * T353[i] for i in range(2, 352)]
Tthird = [(i - 2) * (i - 1) * i * T353[i] for i in range(3, 353)]

rho = 2.955765285651995
c = 0.4399240125710253
rhoinv = 0.33832185689920768


def _eval_T_small_values(x: float) -> float:
    res = 0
    for coeff in reversed(T):
        res = res * x + coeff
    return float(res)


def eval_T(x: float) -> float:
    """
    Computes the value of T(x), for x<rhoinv.

    Note that T(rhoinv) = 1.
    """
    if x < 0.3:
        return _eval_T_small_values(x)
    else:
        L = [_eval_T_small_values(x**k) / k for k in range(2, 50)]
        C = x * np.exp(sum(L))
        p = 0.5
        for _ in range(100):
            p = C * np.exp(p) * (1 - p) / (1 - C * np.exp(p))
        return float(p)


def eval_Tprime(x: float) -> float:
    """
    Computes the value of T'(x), for x<rhoinv**2.
    """
    res = 0
    for coeff in reversed(Tprime):
        res = res * x + coeff
    return float(res)


def eval_Tsecond(x: float) -> float:
    """
    Computes the value of T''(x), for x<rhoinv**2.
    """
    res = 0
    for coeff in reversed(Tsecond):
        res = res * x + coeff
    return float(res)


def eval_Tthird(x: float) -> float:
    """
    Computes the value of T'''(x), for x<rhoinv**2.
    """
    res = 0
    for coeff in reversed(Tthird):
        res = res * x + coeff
    return float(res)


def _ep(x: float) -> float:
    return x * eval_Tprime(x)


def _es(x: float) -> float:
    return (x**2) * eval_Tsecond(x)


def _et(x: float) -> float:
    return (x**3) * eval_Tthird(x)


def expectation_size(x: float, pointed: bool = False) -> float:
    """
    Computes the expected size of the random tree with Boltzmann parameter x.
    """
    A = 1 - eval_T(x)
    B = 1 + sum([_ep(x**k) for k in range(2, 50)])
    if pointed:
        C = sum([k * (_ep(x**k) + _es(x**k)) for k in range(2, 50)])
        return float(B / (A**2) + C / B)
    else:
        return float(B / A)


def variance_size(x: float, pointed: bool = False) -> float:
    """
    Computes the variance of the size of the random tree with Boltzmann
    parameter x.
    """
    A = 1 - eval_T(x)
    B = 1 + sum([_ep(x**k) for k in range(2, 50)])
    C = sum([k * (_ep(x**k) + _es(x**k)) for k in range(2, 50)])
    if pointed:
        D = sum(
            [
                (k**2) * (_ep(x**k) + 3 * _es(x**k) + _et(x**k))
                for k in range(2, 50)
            ]
        )
        res = C / (A**2) + 2 * (B**2) * (1 - A) / (A**4)
        res += D / B - ((C / B) ** 2)
        return res
    else:
        return C / A + (1 - A) * (B**2) / (A**3)


def find_x_for_n(n: int, pointed: bool) -> float:
    """
    Finds the Boltzmann parameter x in order to obtain
    a tree with expected size n.
    """

    def f(x: float):
        return float(expectation_size(x, pointed) - n)

    res, _ = brentq(f, 0.2, rhoinv, full_output=True)
    return float(res)


def compute_values(x: float) -> list[float]:
    """
    Compute the values T(x**k) for k <= 50.
    """
    return [float(0)] + [eval_T(x**k) for k in range(1, 51)]


def sampler_with_precomputed(values, i: int, pointed: bool = False) -> list:
    div_range = range(1, 50 // i + 1)
    N = [0] + [poisson(values[k * i] / k).rvs() for k in div_range]
    res = []
    if i == 1 and pointed:
        P = np.array(values)
        P = P / np.sum(P)
        K = rand.choice(len(P), p=P)
        res += [sampler_with_precomputed(values, K)] * K
    for k in div_range:
        for _ in range(N[k]):
            res += [sampler_with_precomputed(values, k * i)] * k
    return res


def boltzmann_sampler(x: float, pointed: bool = False) -> list:
    """
    Picks at random a rooted unlabelled tree (the result is given as
    a nested list).

    Each tree T has probability x^{|T|} / T(x), where T(x) is the
    generating series of the species of rooted unlabelled trees.

    If the argument pointed is set to True, the random tree has now
    probability |T| x^{|T|} / T.(x), where T.(x) = x T'(x) is the
    generating series of the species of cycle-pointed rooted unlabelled
    trees. This reduces the variance of the size of T.
    """
    values = compute_values(x)
    return sampler_with_precomputed(values, 1, pointed)
