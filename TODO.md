# TODO

programs:

- setuptools?
- rewrite better CrumpJagersModeProcess: right now we can compute up to time T. However, it would be better if we had "live threads" starting from each node of the tree, and if we could just pick the next time of a live thread. Then, the tree has been built up to time T if all live threads are greater than T. Moreover, we can pursue the construction later. But we need to take into account non incrasing birth processes... Idea: define a subclass of RecursiveTree?
- trim trees
- random fragmentation trees
- plot with colors for branching processes?
- [difficult] : animations of Crump-Jagers-Mode


maths:

- expectation and limit law for the profile (at height alpha log n)
- law of a random cut (conditionnally to its size)
