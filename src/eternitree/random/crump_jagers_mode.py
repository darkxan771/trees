# Defines a CrumpJagersModeProcess, which can grow up to a certain time

# TODO: [difficult] animate Crump-Jagers-Mode


from typing import Self

from ..recursive import RecursiveTree
from .point_processes import PointProcess
from .point_processes import repr


class CrumpJagersModeProcess:
    """
    A Crump-Jagers-Mode branching process. It takes as an argument a point
    process pp, such that if i is a node of the tree born at time t, then
    its children are born at times t + X with X ~ pp.

    The associated recursive tree has an additional dictionary
    which contains for each vertex its birth time, and the point
    process of the relative birth times of its children.
    """

    def __init__(self, pp: PointProcess):
        self.pp = pp
        self.pp.reset()
        self.tree = RecursiveTree()
        self.tree.birth_times[0] = float(0)
        self.tree.birth_processes[0] = self.pp.copy()
        t = next(self.tree.birth_processes[0])
        self.to_be_computed = [(t, 0)]

    def __repr__(self):
        res = "CJM branching process with births given by a "
        pp = self.pp
        res += repr[pp.pp_type](pp.parameters)
        return res

    def reset(self) -> None:
        """
        Reset the branching process to its initial state (just the root).
        """
        self.tree = RecursiveTree()
        self.tree.birth_times[0] = float(0)
        self.tree.birth_processes[0] = self.pp.copy()
        t = next(self.tree.birth_processes[0])
        self.to_be_computed = [(t, 0)]

    def grow_up_to_time(self, T: float) -> Self:
        """
        Grows the Crump-Jagers-Mode branching process at least up to time T.
        """
        while not all(
            self.tree.birth_times[x] > T
            or self.tree.birth_processes[x].reaches_time(
                T - self.tree.birth_times[x]
            )
            for x in range(self.tree.size)
        ):
            n = self.tree.size
            t, i = self.to_be_computed.pop(0)
            if self.pp.is_increasing:
                self.tree.add_node(
                    i, birth_time=float(t), birth_process=self.pp.copy()
                )
            else:
                raise NotImplementedError
            u = next(self.tree.birth_processes[i])
            v = next(self.tree.birth_processes[n])
            self.to_be_computed += [
                (u + self.tree.birth_times[i], i),
                (v + self.tree.birth_times[n], n),
            ]
            self.to_be_computed.sort()
        return self

    def compute_tree(self, T: float, reset=True) -> RecursiveTree:
        """
        Computes a sample of the branching process up to time T.
        By default, a new sample is recomputed each time, but setting
        the parameter reset = False keeps the part of the tree which
        has already been computed.
        """
        if reset:
            self.reset()
        _ = self.grow_up_to_time(T)
        R = self.tree
        if max(self.tree.birth_times.values()) > T:
            over = [
                k for k in range(self.tree.size) if self.tree.birth_times[k] > T
            ]
            k = min(over)
            R = R.resize(k)
        return R
