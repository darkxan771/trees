# Defines a CrumpJagersModeProcess, which can grow up to a certain time

from ..recursive import RecursiveTree
from .point_processes import PointProcess


class CrumpJagersModeProcess:
    """
    A Crump-Jagers-Mode branching process. It takes as an argument a point
    process pp, such that if i is a node of the tree born at time t, then
    its children are born at times t + X with X ~ pp.
    """

    def __init__(self, pp: PointProcess):
        self.pp = pp
        self.pp.reset()

    def __repr__(self):
        return f"CJM branching process with births given by a {self.pp}"

    def grow_up_to_time(self, T: float) -> RecursiveTree:
        """
        Samples the Crump-Jagers-Mode branching process until time T.

        The resulting recursive tree has an additional dictionary
        which contains for each vertex its birth time, and the list
        of birth times of its children.
        """
        R = RecursiveTree()
        n = 0
        R.birth_times[0] = float(0)
        R.birth_processes[0] = self.pp.copy()
        _ = R.birth_processes[0].get_random_element(T)
        births = [(t, 0) for t in R.birth_processes[0].state["times"] if t <= T]
        while len(births) > 0:
            t, i = births.pop(0)
            n += 1
            R.add_node(i)
            R.birth_times[n] = float(t)
            R.birth_processes[n] = self.pp.copy()
            R.birth_processes[n].get_random_element(T - t)
            births += [
                (t + R.birth_times[n], n)
                for t in R.birth_processes[n].state["times"]
                if t + R.birth_times[n] <= T
            ]
            births.sort()
        return R
