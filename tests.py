from .recursive_trees import RecursiveTree, RecursiveTrees
from .rooted_trees import RootedTrees
from .random_trees import (
    UniformRootedTree,
    UniformRecursiveTree,
    PlancherelRecursiveTree,
)

import numpy as np


def test_recursive():
    code = [0, 0, 0, 0, 1, 1, 0, 4, 2, 1, 6, 1, 3, 4]
    permutation = [13, 12, 11, 5, 8, 7, 0, 10, 6, 2, 9, 1, 3, 4]
    T = RecursiveTree(max_size=15)
    for i in code:
        T.add_node(i)

    print(T, "\n")

    print(T.first_columns(), "\n")

    T.plot()

    print("code = " + str(T.to_code()))
    print("permutation = " + str(T.to_permutation()))
    print("rooted_tree = " + str(T.to_rooted_tree().to_nested_list()), "\n")

    print("edges = " + str(T.number_of_edges()))
    print("vertices = " + str(T.number_of_vertices()))
    print("height = " + str(T.height()))
    print("profile = " + str(T.profile()))
    print("degrees = " + str(T.degrees()), "\n")

    print("subtree from node 1 = " + str(T.subtree_indices(1)))

    print("subtree = ")
    T.subtree(1).plot()
    print("\n")

    print("cut = ")
    T.cut(1).plot()
    print("\n")

    T.plot_distribution("degree")
    T.plot_distribution("depth")
    T.plot_distribution("size")
    print("\n")

    Rec = RecursiveTrees(15)
    print(Rec)
    print("cardinality = " + str(Rec.cardinality()))
    print(T == Rec.from_code(code))
    print(T == Rec.from_permutation(permutation))


def test_rooted():
    code = [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 5, 4, 5, 4, 3]
    nested = [[[[[[[[]], []], [], []], [[]], []], []]]]
    T = RootedTrees(15).from_nested_list(nested)

    print(T, "\n")

    T.plot()

    print("code = " + str(T.to_code()))
    print("edges = " + str(T.number_of_edges()))
    print("vertices = " + str(T.number_of_vertices()))
    print("height = " + str(T.height()))
    print("d = " + str(T.d()))
    print("u = " + str(T.u()))
    print("sym = " + str(T.sym()))
    print("\n")

    Root = RootedTrees(15)
    print("cardinality = " + str(Root.cardinality()))
    print(T == Root.from_code(code))


def test_random():
    print("Random uniform rooted tree")
    U = UniformRootedTree(30).get_random_element()
    U.plot(style="circular", with_circles=True, node_size=50)
    print("\n")

    print("Random uniform recursive tree")
    T = UniformRecursiveTree(30).get_random_element()
    T.plot()
    print("\n")

    print("Random uniform Plancherel tree")
    PT = PlancherelRecursiveTree(30).get_random_element()
    PT.plot()


def test_subtree_plancherel(N, K):
    """
    Tests if, conditionally to being of size K, a random subtree
    of a Plancherel recursive tree with order N is also
    Plancherel distributed.
    """
    nums = {T.to_code(): float(0) for T in RecursiveTrees(K)}
    dplancherel = {
        T.to_code(): PlancherelRecursiveTree(K).probability(T)
        for T in RecursiveTrees(K)
    }
    denom = float(0)
    for U in RecursiveTrees(N):
        plan = PlancherelRecursiveTree(N).probability(U)
        denom += float(plan * np.count_nonzero(U.size[:N] == K))
        for k in range(N):
            if U.size[k] == K:
                nums[U.subtree(k).to_code()] += plan
    d = {code: nums[code] / denom for code in nums}
    return all(np.isclose(d[c], dplancherel[c]) for c in d)
