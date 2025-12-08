# Defines an abstract class PointProcess, and various subclasses

from __future__ import annotations

from typing import Callable, Self

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scs
from matplotlib.patches import Circle
from scipy.integrate import quad
from scipy.optimize import brentq


def inverse_function(F: Callable[[float], float]) -> Callable[[float], float]:
    def inv(x: float) -> float:
        return brentq(lambda t: F(t) - x, 0, 1e6, full_output=True)[0]

    return inv


def EXP():
    return float(scs.expon(scale=1).rvs())


def BETA1(theta: float) -> float:
    return float(scs.beta(1, theta).rvs())


class StatePointProcess:
    """
    A list of parameters which determine the state of a point process.
    """

    def __init__(self, d: dict = {"times": []}):
        if "times" not in d:
            raise ValueError(
                "The state should be initialized with a list of times."
            )
        self.data = d.copy()
        self.lists = []
        self.nums = []
        for key, value in d.items():
            if isinstance(value, list):
                self.lists.append(key)
            else:
                self.nums.append(key)

    def __repr__(self):
        return self.data.__repr__()

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, item):
        self.data[key] = item

    def reset(self) -> Self:
        for key in self.lists:
            self.data[key] = []
        for key in self.nums:
            self.data[key] = float(0)
        return self


class PointProcess:
    """
    A point process on R+. It is initialized with a state and
    a function next_time, which takes as argument the list of
    previously computed times and returns the next random time.

    Note: "next" does not mean in general that the indexing
    of times is increasing.
    """

    def __init__(
        self,
        next_time: Callable[[list], float] = (lambda T: ([0] + T)[-1] + EXP()),
        state: StatePointProcess = StatePointProcess(),
    ):
        self.state = state
        self.next_time: Callable[[list], float] = next_time
        self.name_prepend: str = "Poisson"
        self.name_append: str = "on R+"
        self.is_increasing: bool = True

    def __repr__(self):
        res = f"{self.name_prepend} point process {self.name_append}"
        res += f" | times = {self.state.data["times"]}"
        return res

    def __iter__(self):
        return self

    def __next__(self):
        self.state["times"].append(self.next_time(self.state["times"]))
        return self.state["times"][-1]

    def reset(self) -> Self:
        _ = self.state.reset()
        return self

    def copy(self) -> PointProcess:
        """
        Returns a new copy of the point process.
        """
        state = StatePointProcess(self.state.data)
        state.reset()
        res = PointProcess(self.next_time, state)
        res.name_prepend, res.name_append, res.is_increasing = (
            self.name_prepend,
            self.name_append,
            self.is_increasing,
        )
        return res

    def reach_time(self, T: float) -> None:
        """
        Ensures that the point process has been completely
        computed on [0, T].
        """
        if self.is_increasing:
            while (self.state["times"] == []) or (self.state["times"][-1] < T):
                _ = next(self)
        else:
            raise NotImplementedError

    def get_random_element(self, T: float) -> np.ndarray:
        """
        Samples the point process on the interval [0,T].
        """
        self.state.reset()
        self.reach_time(T)
        return np.array([t for t in self.state["times"] if t <= T])

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

    def __init__(
        self,
        density: Callable[[float], float] = (lambda _: 1),
        state: StatePointProcess = StatePointProcess(),
    ):
        self.F = lambda x: quad(density, 0, x)[0]
        self.inv = inverse_function(self.F)
        self.state = state
        self.next_time = lambda L: self.inv(self.F(([0] + L)[-1]) + EXP())
        self.name_prepend = "Poisson"
        self.name_append = "on R+"
        self.is_increasing = True


class PoissonDirichlet(PointProcess):
    """
    A Poisson Dirichlet point process on [0,1] with parameter theta.
    """

    def __init__(self, theta: float = 1):
        self.theta = theta
        self.state = StatePointProcess({"times": [], "beta": [], "cprod": 0})
        self.next_time = lambda T: ([0] + T)[-1] + (
            1 - self.state["cprod"]
        ) * BETA1(self.theta)
        self.name_prepend = "Poisson Dirichlet"
        self.name_append = f"with parameter {self.theta}"
        self.is_increasing = False

    def __next__(self):
        self.state["beta"].append(BETA1(self.theta))
        self.state["times"].append(
            ([0] + self.state["times"])[-1]
            + (1 - self.state["cprod"]) * self.state["beta"][-1]
        )
        self.state["cprod"] = 1 - (1 - self.state["beta"][-1]) * (
            1 - self.state["cprod"]
        )
        return self.state["times"][-1]

    def reach_time(self, T: float) -> None:
        """
        Ensures that the point process has been completely
        computed on [0, T].
        """
        while T > self.state["cprod"]:
            _ = next(self)


class LogPoissonDirichlet(PointProcess):
    """
    Beware! We take the logarithms of the blocks, not the times of PD(theta).
    Therefore, the times are not increasing in general. However, we can know
    when we have all the points in a interval [0,T]
    """

    def __init__(self, theta: float = 1):
        self.theta = theta
        self.state = StatePointProcess({"times": [], "beta": [], "cprod": 0})
        self.next_time = lambda _: -np.log(
            (1 - self.state["cprod"]) * BETA1(self.theta)
        )
        self.name_prepend = "Log Poisson Dirichlet"
        self.name_append = f"with parameter {self.theta}"
        self.is_increasing = False

    def __next__(self):
        self.state["beta"].append(BETA1(self.theta))
        self.state["times"].append(
            -np.log((1 - self.state["cprod"]) * self.state["beta"][-1])
        )
        self.state["cprod"] = 1 - (1 - self.state["cprod"]) * (
            1 - self.state["beta"][-1]
        )
        return self.state["times"][-1]

    def reach_time(self, T: float) -> None:
        """
        Ensures that the point process has been completely
        computed on [0, T].
        """
        while T > -np.log(1 - self.state["cprod"]):
            _ = next(self)
