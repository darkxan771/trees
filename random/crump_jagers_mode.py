# Defines a CrumpJagersModeProcess
# It can be constructed up to a certain time, or up to a certain size

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

    def __repr__(self):
        return f"CJM branching process with births given by a {self.pp}"

    def __call__(self, N: int):
        from .random_trees import CrumpJagersModeTree

        return CrumpJagersModeTree(N, self.pp)

    def grow_up_to_time(self, T: float) -> RecursiveTree:
        """
        Samples the Crump-Jagers-Mode branching process until time T.
        """
        self.pp.reset()
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
