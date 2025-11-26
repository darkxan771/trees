import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from scipy.optimize import brentq
from scipy.integrate import quad
from typing import Callable
from ..recursive import RecursiveTree


def inverse_function(F: Callable[[float], float]) -> Callable[[float], float]:
    def inv(x: float) -> float:
        return brentq(lambda t: F(t) - x, 0, 1e6, full_output=True)[0]

    return inv


def EXP():
    return scs.expon(scale=1).rvs()


class PointProcess:
    """
    A point process on R+. It is initialized with a function
    next_time, which takes as argument the list of previously
    computed times and returns the next random time.
    """

    def __init__(
        self,
        next_time: Callable[[list], float] = (lambda T: ([0] + T)[-1] + EXP()),
        n_prep: str = "Standard Poisson",
        n_app: str = "",
    ):
        self.next_time = next_time
        self.times = []
        self.name_prepend = n_prep
        self.name_append = n_app

    def __repr__(self):
        return f"{self.name_prepend} point process on R+ {self.name_append}"

    def __iter__(self):
        return self

    def __next__(self):
        self.times.append(self.next_time(self.times))
        return self.times[-1]

    def get_random_element(self, T: float) -> np.ndarray:
        """
        Samples the point process on the interval [0,T].
        """
        self.times = []
        while (self.times == []) or (self.times[-1] < T):
            _ = next(self)
        return np.array(self.times[:-1])

    def plot(self, T: float) -> None:
        """
        Plots a sample of the point process on the interval [0,T].
        """
        data = self.get_random_element(T)
        _, ax = plt.subplots()
        ax.plot([0, T], [0, 0], color="k")
        for t in data:
            ax.add_patch(Circle((t, 0), 0.1, fill=True, color="b", zorder=3))
        ax.set_xticks([0, T])
        ax.set_xlim(0, T)
        ax.set_ylim(-1, 1)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect(1)
        ax.set_axisbelow(True)
        plt.show()


class PoissonPointProcess(PointProcess):
    """
    A Poisson point process with intensity f(x) dx.
    """

    def __init__(self, f: Callable[[float], float] = (lambda x: 1)):
        self.f = f
        self.F = lambda x: quad(f, 0, x)[0]
        self.inv = inverse_function(self.F)
        self.next_time = lambda L: self.inv(self.F(([0] + L)[-1]) + EXP())
        self.times = []
        self.name_prepend = "Poisson"
        self.name_append = ""


class CrumpJagersModeProcess:
    """
    A Crump-Jagers-Mode branching process. It takes as an argument a point
    process pp, such that if i is a node of the tree born at time t, then
    its children are born at times t + X with X ~ pp.
    """

    def __init__(self, pp: PointProcess):
        self.pp = pp

    def __repr__(self):
        return f"CJM branching process with births given by a {self.pp}"

    def __call__(self, N: int):
        from .random_trees import CrumpJagersModeTree

        return CrumpJagersModeTree(N, self.pp)

    def grow_up_to_time(self, T: float) -> RecursiveTree:
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
        Samples the Crump-Jagers-Mode branching process until
        it reaches a size N.
        """
        T = self(N).get_random_element()
        if isinstance(T, RecursiveTree):
            return T
        else:
            raise ValueError
