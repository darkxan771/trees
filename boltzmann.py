import numpy as np
from scipy.optimize import brentq


def generating_series_T(N: int):
    """
    Computes the N first terms of the generating series of
    rooted unlabelled trees.
    """
    divs = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            divs[j].append(i)
    res = [0] * (N + 1)
    res2 = [0] * (N + 1)
    res[1] = 1
    res2[1] = 1
    for n in range(1, N):
        res[n + 1] = sum([res[n - k] * res2[k + 1] for k in range(n)]) // n
        res2[n + 1] = sum([d * res[d] for d in divs[n + 1]])
    return res


# T5000 = generating_series_T(5000)
# should take around 7 seconds.


T353 = generating_series_T(353)
T = T353[:350]
Tprime = [i * T353[i] for i in range(1, 351)]
Tsecond = [(i - 1) * i * T353[i] for i in range(2, 352)]
Tthird = [(i - 2) * (i - 1) * i * T353[i] for i in range(3, 353)]

rho = 2.955765285651995
c = 0.4399240125710253
rhoinv = 0.33832185689920768


def _eval_T_small_values(x: float):
    res = 0
    for coeff in reversed(T):
        res = res * x + coeff
    return float(res)


def eval_T(x: float):
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


def eval_Tprime(x: float):
    """
    Computes the value of T'(x), for x<rhoinv**2.
    """
    res = 0
    for coeff in reversed(Tprime):
        res = res * x + coeff
    return float(res)


def eval_Tsecond(x: float):
    """
    Computes the value of T''(x), for x<rhoinv**2.
    """
    res = 0
    for coeff in reversed(Tsecond):
        res = res * x + coeff
    return float(res)


def eval_Tthird(x: float):
    """
    Computes the value of T'''(x), for x<rhoinv**2.
    """
    res = 0
    for coeff in reversed(Tthird):
        res = res * x + coeff
    return float(res)


def _ep(x: float):
    return x * eval_Tprime(x)


def _es(x: float):
    return (x**2) * eval_Tsecond(x)


def _et(x: float):
    return (x**3) * eval_Tthird(x)


def expectation_size(x: float, pointed: bool = False):
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


def variance_size(x: float, pointed: bool = False):
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


def find_x_for_n(n: int, pointed: bool):
    """
    Finds the Boltzmann parameter x in order to obtain
    a tree with expected size n.
    """

    def f(x: float):
        return float(expectation_size(x, pointed) - n)

    res, _ = brentq(f, 0.2, rhoinv, full_output=True)
    return float(res)


def compute_values(x: float):
    """
    Compute the values T(x**k) for k <= 50.
    """
    return [0] + [eval_T(x**k) for k in range(1, 51)]
