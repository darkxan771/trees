# Defines an abstract class PointProcess, and various subclasses

import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from scipy.optimize import brentq
from scipy.integrate import quad
from typing import Callable


def inverse_function(F: Callable[[float], float]) -> Callable[[float], float]:
    def inv(x: float) -> float:
        return brentq(lambda t: F(t) - x, 0, 1e6, full_output=True)[0]

    return inv


def EXP():
    return float(scs.expon(scale=1).rvs())


def extract_beta(L: list) -> list:
    res = []
    prod = 1
    for t in L:
        res.append(1 - (1 - t) / prod)
        prod *= 1 - res[-1]
    return res


def BETA1(theta: float) -> float:
    return float(scs.beta(1, theta).rvs())


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
        n_app: str = "on R+",
    ):
        self.times = []
        self.next_time = next_time
        self.name_prepend = n_prep
        self.name_append = n_app

    def __repr__(self):
        return f"{self.name_prepend} point process {self.name_append}"

    def __iter__(self):
        return self

    def __next__(self):
        self.times.append(self.next_time(self.times))
        return self.times[-1]

    def reset(self):
        self.times = []

    def get_random_element(self, T: float) -> np.ndarray:
        """
        Samples the point process on the interval [0,T].
        """
        self.reset()
        while (self.times == []) or (self.times[-1] < T):
            _ = next(self)
        return np.array(self.times[:-1])

    def plot(self, T: float, size: float = 1) -> None:
        """
        Plots a sample of the point process on the interval [0,T].
        """
        data = self.get_random_element(T)
        _, ax = plt.subplots()
        ax.plot([0, T], [0, 0], color="k")
        for t in data:
            ax.add_patch(
                Circle((t, 0), 0.1 * size, fill=True, color="b", zorder=3)
            )
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

    def __init__(self, f: Callable[[float], float] = (lambda _: 1)):
        self.times = []
        self.f = f
        self.F = lambda x: quad(f, 0, x)[0]
        self.inv = inverse_function(self.F)
        self.next_time = lambda L: self.inv(self.F(([0] + L)[-1]) + EXP())
        self.name_prepend = "Poisson"
        self.name_append = "on R+"


class PoissonDirichlet(PointProcess):
    def __init__(self, theta: float):
        self.theta = theta
        self.times = []
        self.next_time = lambda L: float(
            np.prod(1 - np.array(extract_beta(L))) * BETA1(self.theta)
        )
        self.beta = []
        self.prod = 1
        self.name_prepend = "Poisson Dirichlet"
        self.name_append = f"with parameter {self.theta}"

    def __next__(self):
        self.beta.append(BETA1(self.theta))
        self.times.append(([0] + self.times)[-1] + self.beta[-1] * self.prod)
        self.prod *= 1 - self.beta[-1]
        return self.times[-1]

    def reset(self):
        self.times = []
        self.beta = []
        self.prod = 1


class LogPoissonDirichlet(PointProcess):
    """
    Beware! We take the logarithms of the blocks, not the times of PD(theta).
    Therefore, the times are not increasing in general. However, we can know
    when we have all the points in a interval [0,T]
    """

    def __init__(self, theta: float):
        self.theta = theta
        self.times = []
        self.beta = []
        self.prod = 1
        self.next_time = lambda L: 0
        self.name_prepend = "Log Poisson Dirichlet"
        self.name_append = f"with parameter {self.theta}"

    def __next__(self):
        self.beta.append(BETA1(self.theta))
        self.times.append(-np.log(self.beta[-1] * self.prod))
        self.prod *= 1 - self.beta[-1]
        return self.times[-1]

    def reset(self):
        self.times = []
        self.beta = []
        self.prod = 1

    def get_random_element(self, T: float) -> np.ndarray:
        """
        Samples the point process on the interval [0,T].
        """
        self.reset()
        while -np.log(self.prod) <= T:
            _ = next(self)
        return np.array([x for x in self.times if x <= T])
