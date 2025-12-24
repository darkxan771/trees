# Tests

import sys

import numpy as np

sys.path.insert(0, "../src/")

from eternitree.abstraction import IntegerPartition
from eternitree.containers import IntegerPartitions
from eternitree.containers import RecursiveTrees
from eternitree.containers import RootedTrees
from eternitree.random import PlancherelPartition
from eternitree.random import PlancherelRecursiveTree
from eternitree.random import UniformPartition
from eternitree.random import UniformRecursiveTree
from eternitree.random import UniformRootedTree
from eternitree.recursive import RecursiveTree


def test_recursive(verbose=True):
    code = (0, 0, 0, 0, 1, 1, 0, 4, 2, 1, 6, 1, 3, 4)
    permutation = (13, 12, 11, 5, 8, 7, 0, 10, 6, 2, 9, 1, 3, 4)
    L = [[[[]], [], [], []], [[], []], [[]], [[]], []]
    T = RecursiveTree(max_size=15)
    for i in code:
        _ = T.add_node(i)

    assert T.convert("code") == code
    assert T.convert("permutation") == permutation
    assert T.convert("rooted").convert("nested_list") == L
    assert T.number_of_edges == 14
    assert T.number_of_vertices == 15
    assert T.height == 3
    assert T.children == [
        [1, 2, 3, 4, 7],
        [5, 6, 10, 12],
        [9],
        [13],
        [8, 14],
        [],
        [11],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    assert T.profile.tolist() == [1, 5, 8, 1]
    assert T.subtrees_partition.parts == [6, 3, 2, 2, 1]
    assert T.path_to_root(6) == [0, 1, 6]
    assert T.longest_path == [0, 1, 6, 11]
    assert T.row_positions.tolist() == [
        0,
        0,
        1,
        2,
        3,
        0,
        1,
        4,
        6,
        4,
        2,
        0,
        3,
        5,
        7,
    ]

    degrees = np.array([3 / 5, 1 / 5, 1 / 15, 0, 1 / 15, 1 / 15])
    assert np.all(np.isclose(T.statistic("degree").array, degrees))
    depths = np.array([1 / 15, 1 / 3, 8 / 15, 1 / 15])
    assert np.all(np.isclose(T.statistic("depth").array, depths))
    sizes = np.array(
        [0, 3 / 5, 1 / 5, 1 / 15, 0, 0, 1 / 15, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 15]
    )
    assert np.all(np.isclose(T.statistic("size").array, sizes))
    assert T.subtree_indices(1) == [1, 5, 6, 10, 11, 12]

    S = RecursiveTree(max_size=6)
    for j in (0, 0, 0, 2, 0):
        _ = S.add_node(j)
    assert T.subtree(1) == S

    C = RecursiveTree(max_size=12)
    for k in (0, 0, 0, 1, 1, 0, 2, 1, 5, 1, 3):
        _ = C.add_node(k)
    assert T.cut(T.subtree_indices(4)) == C

    T2 = RecursiveTrees().example()
    assert T2 == T
    T2.convert("recursive").insert_node(10, 2)
    assert T2.convert("code") == (0, 0, 0, 0, 1, 1, 0, 4, 2, 2, 1, 6, 1, 3, 4)

    T.weight = np.array([15, 13, 14, 6, 5, 9, 12, 11, 2, 8, 7, 10, 3, 1, 4])
    assert T.is_double_recursive()
    KP = np.array(
        [
            [1, 2, 3, 4, 3, 4, 7, 1, 8, 8, 8, 10, 4, 4],
            [1, 2, 1, 1, 3, 4, 4, 1, 4, 4, 7, 2, 1, 4],
        ]
    )
    assert np.all(T.convert("KP_insertion_array") == KP)

    Rec = RecursiveTrees(15)
    assert T in Rec
    assert Rec.cardinality == 87178291200
    assert Rec.from_permutation(permutation) == T
    assert Rec.from_code(code) == T
    assert Rec.from_KP_insertion_array(KP) == T

    if verbose:
        print("\n", "Done !")
        print(T.convert("dataframe"))
        T.plot()


def test_rooted(verbose=True):
    Root = RootedTrees(15)
    code = (0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 5, 4, 5, 4, 3)
    nested = [[[[[[[[]], []], [], []], [[]], []], []]]]
    T = RootedTrees(15).from_nested_list(nested)

    assert T.convert("code") == code
    assert T.convert("nested_list") == nested
    assert T.number_of_edges == 14
    assert T.number_of_vertices == 15
    assert T.height == 7
    assert T.profile.tolist() == [1, 1, 1, 2, 3, 4, 2, 1]
    assert T.subtrees_partition.parts == [14]
    assert T.d == 388800
    assert T.u == 194400
    assert T.sym == 2
    assert Root.from_code(code) == T
    assert Root.cardinality == 87811

    if verbose:
        T.plot()
        print("\n", "Done !")


def test_partition(verbose=True):
    P = IntegerPartition([5, 3, 2, 2])
    P12 = IntegerPartitions(12)

    assert len(list(P12)) == P12.cardinality == 77
    assert P in P12
    assert P.size == 12
    assert P.length == 4
    assert P.dictionary == {2: 2, 3: 1, 5: 1}
    assert P.bell_number == 83160
    assert P.z == 120
    assert P.conjugate == IntegerPartition([4, 4, 2, 1, 1])
    assert P.dimension == 4455
    assert [P.parts for P in IntegerPartitions(4)] == [
        [1, 1, 1, 1],
        [2, 1, 1],
        [3, 1],
        [2, 2],
        [4],
    ]

    if verbose:
        P.plot()
        print("\n", "Done !")


def test_random(verbose=True):
    U = UniformRootedTree(30).get_random_element()
    UR = UniformRootedTree(5)
    R = RootedTrees(5)
    assert all(np.isclose(UR.distribution()[T.code], 1 / 9) for T in R)

    T = UniformRecursiveTree(30).get_random_element()
    UR2 = UniformRecursiveTree(5)
    R2 = RecursiveTrees(5)
    assert all(
        np.isclose(UR2.distribution()[T.convert("code")], 1 / 24) for T in R2
    )

    PT = PlancherelRecursiveTree(30).get_random_element()

    P = UniformPartition(30).get_random_element()
    print("\n")
    UP12 = UniformPartition(12)
    P12 = IntegerPartitions(12)
    assert all(
        np.isclose(UP12.distribution()[tuple(P.parts)], 1 / 77) for P in P12
    )

    PP = PlancherelPartition(100).get_random_element()
    if verbose:
        print("Random uniform rooted tree", "\n")
        U.plot(style="circular", with_levels=True, node_size=50)
        print("\n", "Random uniform recursive tree", "\n")
        T.plot()
        print("\n", "Random Plancherel recursive tree", "\n")
        PT.plot()
        print("\n", "Random uniform partition", "\n")
        P.plot()
        print("\n", "Random Plancherel partition", "\n")
        PP.plot()
        print("\n", "Done !")


def main():
    test_recursive(False)
    test_rooted(False)
    test_partition(False)
    test_random(False)
    print("Done !")


if __name__ == "__main__":
    main()
