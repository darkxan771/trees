import numpy as np
import numpy.random as rand
from scipy.stats import poisson


def print_graphic_options():
    graphic_options = """
        node_size (int, 300)
        node_shape (str, "o")
        arrows (bool, True)
        arrow_size (int, 10)
        width (float, 1.0)
        node_color, edge_color
        font_size (int, 12)
        """
    print(graphic_options)


def code_flatten(data):
    for x in data:
        yield from x


def raise_tuple(t: tuple):
    return tuple(x+1 for x in t)


def nested_list_to_code(L):
    return tuple(code_flatten([(0,)]
                 + [raise_tuple(nested_list_to_code(c)) for c in L]))


def code_to_nested_list(L):
    res = []
    if (not (L[0] == 0)):
        raise ValueError
    if len(L) > 1:
        child = [0]
        for i in range(2, len(L)):
            if L[i] == 1:
                res.append(code_to_nested_list(child))
                child = [0]
            else:
                child.append(L[i] - 1)
        res.append(code_to_nested_list(child))
    return res


def standardisation(L):
    n = L.size
    res = np.ones(n, dtype=int)
    for k in range(1, n):
        for j in range(k):
            res[j] += int(L[j] > L[k])
            res[k] += int(L[j] < L[k])
    return res


def random_pairs(n: int):
    alea_w = (np.arange(1, n) + np.arange(1, n)**2) * rand.random(size=n-1)
    w = np.floor(np.sqrt(alea_w + 0.25) + 0.5)
    J = 1 + rand.randint(w)
    return (w.astype(int), J)


def permutation_to_code(p: np.ndarray):
    """
    Converts a permutation array to a code array.
    """
    res = np.zeros(p.size, dtype=int)
    for i in range(1, len(p)):
        res[i] = np.count_nonzero(p[:i] < p[i])
    return res


def code_to_permutation(c: np.ndarray):
    """
    Converts a code array to a permutation array.
    """
    res = np.zeros(c.size, dtype=int)
    for i in range(1, len(c)):
        res[i] = c[i]
        res[:i] += (res[:i] >= res[i])
    return res


def poisson_galton_watson(mu: float):
    """
    Returns the data required to create a Galton-Watson tree
    with offspring distribution Poisson(mu).
    """
    xi = poisson(mu).rvs()
    subs = [poisson_galton_watson(mu) for _ in range(xi)]
    size = 1 + sum([c[0] for c in subs])
    return size, subs
