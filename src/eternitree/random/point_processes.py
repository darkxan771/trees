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
        """
        Empty the state of the point process.
        """
        for key in self.lists:
            self.data[key] = []
        for key in self.nums:
            self.data[key] = float(0)
        return self


##################
# Repr functions #
##################


def repr_poisson(args: dict) -> str:
    if "intensity" in args.keys():
        return f"Poisson point process with intensity {args["intensity"]}"
    elif "density" in args.keys():
        return f"Non-homogeneous Poisson point process"
    else:
        raise NotImplementedError


def repr_PD(args) -> str:
    return f"Poisson-Dirichlet point process with parameter {args["theta"]}"


def repr_LPD(args) -> str:
    return f"Log-Poisson-Dirichlet point process with parameter {args["theta"]}"


repr: dict[str, Callable] = {
    "Poisson": repr_poisson,
    "Poisson-Dirichlet": repr_PD,
    "Log-Poisson-Dirichlet": repr_LPD,
}


##################
# Next functions #
##################


def next_time_poisson(state: StatePointProcess, args: dict) -> float:
    if "intensity" in args.keys():
        L = args["intensity"]
        t = ([0] + state["times"])[-1] + EXP() / L
    elif "density" in args.keys():
        f = args["density"]
        F = lambda x: quad(f, 0, x)[0]
        inv = inverse_function(F)
        t = inv(F(([0] + state["times"])[-1]) + EXP())
    else:
        raise NotImplementedError
    state["times"].append(t)
    return t


def next_time_PD(state: StatePointProcess, args: dict) -> float:
    state["beta"].append(BETA1(args["theta"]))
    state["times"].append(
        ([0] + state["times"])[-1] + (1 - state["cprod"]) * state["beta"][-1]
    )
    state["cprod"] = 1 - (1 - state["beta"][-1]) * (1 - state["cprod"])
    return state["times"][-1]


def next_time_LPD(state: StatePointProcess, args: dict) -> float:
    state["beta"].append(BETA1(args["theta"]))
    state["times"].append(-np.log((1 - state["cprod"]) * state["beta"][-1]))
    state["cprod"] = 1 - (1 - state["beta"][-1]) * (1 - state["cprod"])
    return state["times"][-1]


next_time: dict[str, Callable] = {
    "Poisson": next_time_poisson,
    "Poisson-Dirichlet": next_time_PD,
    "Log-Poisson-Dirichlet": next_time_LPD,
}


###################
# Reach functions #
###################


def reaches_poisson(state: StatePointProcess, T: float) -> bool:
    return (state["times"] != []) and (state["times"][-1] >= T)


def reaches_PD(state: StatePointProcess, T: float) -> bool:
    return T <= state["cprod"]


def reaches_LPD(state: StatePointProcess, T: float) -> bool:
    return T <= -np.log(1 - state["cprod"])


reach: dict[str, Callable] = {
    "Poisson": reaches_poisson,
    "Poisson-Dirichlet": reaches_PD,
    "Log-Poisson-Dirichlet": reaches_LPD,
}


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
        pp_type: str = "Poisson",
        state: StatePointProcess = StatePointProcess(),
        **args,
    ):
        self.pp_type = pp_type
        self.state = state
        self.parameters = args

    def __repr__(self):
        res = repr[self.pp_type](self.parameters)
        res += f" | times = {self.state.data["times"]}"
        return res

    def __iter__(self):
        return self

    def __next__(self):
        return next_time[self.pp_type](self.state, self.parameters)

    def reset(self) -> Self:
        """
        Reset the point process to its initial state.
        """
        _ = self.state.reset()
        return self

    @property
    def is_increasing(self) -> bool:
        """
        Indicates whether the point process produces its point
        in an increasing manner.
        """
        increasing = ["Poisson"]
        return self.pp_type in increasing

    def copy(self) -> PointProcess:
        """
        Returns a new copy of the point process.
        """
        state = StatePointProcess(self.state.data)
        state.reset()
        res = PointProcess(self.pp_type, state, **self.parameters)
        return res

    def reaches_time(self, T: float) -> bool:
        """
        Checks if the point process has been completely
        computed on [0, T].
        """
        return reach[self.pp_type](self.state, T)

    def compute_up_to_time(self, T: float) -> None:
        """
        Computes the point process completely on [0, T].
        """
        while not self.reaches_time(T):
            _ = next(self)

    def get_random_element(self, T: float) -> np.ndarray:
        """
        Samples the point process on the interval [0,T].
        """
        self.state.reset()
        self.compute_up_to_time(T)
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


def PoissonPointProcess() -> PointProcess:
    return PointProcess("Poisson", intensity=1)


def PoissonDirichlet(theta: float) -> PointProcess:
    return PointProcess(
        "Poisson-Dirichlet",
        state=StatePointProcess({"times": [], "beta": [], "cprod": 0}),
        theta=theta,
    )


def LogPoissonDirichlet(theta: float) -> PointProcess:
    return PointProcess(
        "Log-Poisson-Dirichlet",
        state=StatePointProcess({"times": [], "beta": [], "cprod": 0}),
        theta=theta,
    )
