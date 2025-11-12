import numpy as np
import scipy.stats as scs

from copy import deepcopy
from .recursive_trees import RecursiveTree


class PointProcess:
    """
    A point process on R+. It is initialized with a function
    next_time, which takes as argument the list of previously
    computed times and returns the next random time.
    """

    def __init__(self, next_time, n_prep: str = "", n_app: str = ""):
        self.next = next_time
        self.times = []
        self.name_prepend = n_prep
        self.name_append = n_app

    def __repr__(self):
        return f"{self.name_prepend} Point process on R+ {self.name_append}"

    def __iter__(self):
        return self

    def __next__(self):
        self.times.append(self.next(self.times))
        return self.times[-1]

    def get_random_element(self, T: float) -> np.ndarray:
        """
        Samples the point process on the interval [0,T].
        """
        self.times = []
        while (self.times == []) or (self.times[-1] < T):
            _ = next(self)
        return np.array(self.times[:-1])


def PoissonPointProcess(L: float = 1) -> PointProcess:
    """
    Returns a Poisson point process with intensity L.
    """
    return PointProcess(
        lambda times: ([0] + times)[-1] + scs.expon(scale=1 / L).rvs(),
        n_prep="Poisson",
        n_app=f"with intensity {L}",
    )


class CrumpJagersMode:
    """
    A Crump-Jagers-Mode branching process. It takes as an argument a point
    process pp, such that if i is a node of the tree born at time t, then
    its children are born at times t + X with X ~ pp.
    """

    def __init__(self, pp: PointProcess):
        self.pp = pp

    def __repr__(self):
        return f"CJM branching process with births given by a {self.pp}"

    def get_random_element(self, T: float) -> RecursiveTree:
        """
        Samples the Crump-Jagers-Mode branching process until time T.
        """
        R = RecursiveTree()
        n = 0
        d = {0: self.pp.get_random_element(T)}
        births = [(t, 0) for t in d[0]]
        while len(births) > 0:
            t, i = births.pop(0)
            n += 1
            R.add_node(i)
            d[n] = t + self.pp.get_random_element(T - t)
            births += [(t, n) for t in d[n]]
            births.sort()
        R.additional = d
        return R

    def grow_up_to_size(self, N: int) -> RecursiveTree:
        """
        Samples the Crump-Jagers-Mode branching process until one obtains
        a recursive tree with size N.
        """
        R = RecursiveTree(max_size=N)
        Lpp = [deepcopy(self.pp)]
        birth = [0]
        Lpp[0].times = []
        next_computed = []
        n = 1
        while n < N:
            for i in range(n):
                if i not in [x[0] for x in next_computed]:
                    next_computed.append((i, birth[i] + next(Lpp[i])))
            next_computed.sort(key=lambda x: x[1])
            i, t = next_computed.pop(0)
            R.add_node(i)
            n += 1
            birth.append(t)
            Lpp.append(deepcopy(self.pp))
            Lpp[-1].times = []
        last = birth[N - 1]
        for i in range(N):
            L = [birth[i] + t for t in Lpp[i].times if birth[i] + t < last]
            R.additional[i] = np.array(L)
        return R
