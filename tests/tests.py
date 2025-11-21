import numpy as np

from ..recursive.recursive_tree import RecursiveTree
from ..containers.trees import RootedTrees, RecursiveTrees
from ..random.random_trees import (
    UniformRootedTree,
    UniformRecursiveTree,
    PlancherelRecursiveTree,
)
from ..abstraction.partition import (
    IntegerPartition,
)
from ..containers.partitions import IntegerPartitions


def test_recursive():
    code = (0, 0, 0, 0, 1, 1, 0, 4, 2, 1, 6, 1, 3, 4)
    permutation = (13, 12, 11, 5, 8, 7, 0, 10, 6, 2, 9, 1, 3, 4)
    L = [[[[]], [], [], []], [[], []], [[]], [[]], []]
    T = RecursiveTree(max_size=15)
    for i in code:
        _ = T.add_node(i)

    assert T.convert("code") == code
    assert T.convert("permutation") == permutation
    assert T.convert("rooted").convert("nested_list") == L

    print(T.convert("dataframe"))

    T.plot()

    assert T.number_of_edges == 14
    assert T.number_of_vertices == 15
    assert T.height == 3
    assert T.profile.tolist() == [1, 5, 8, 1]
    assert T.subtrees_partition.parts == [6, 3, 2, 2, 1]
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

    assert T.path_to_root(13).tolist() == [0, 3, 13]
    assert T.subtree_indices(1) == [1, 5, 6, 10, 11, 12]

    S = RecursiveTree(max_size=6)
    for j in (0, 0, 0, 2, 0):
        _ = S.add_node(j)
    assert T.subtree(1) == S

    C = RecursiveTree(max_size=12)
    for k in (0, 0, 0, 1, 1, 0, 2, 1, 5, 1, 3):
        _ = C.add_node(k)
    assert T.cut(4) == C

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
    assert Rec.cardinality() == 87178291200
    assert Rec.from_permutation(permutation) == T
    assert Rec.from_code(code) == T
    assert Rec.from_KP_insertion_array(KP) == T

    print("\n", "Done !")


def test_rooted():
    Root = RootedTrees(15)
    code = (0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 5, 4, 5, 4, 3)
    nested = [[[[[[[[]], []], [], []], [[]], []], []]]]
    T = RootedTrees(15).from_nested_list(nested)

    T.plot()

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
    assert Root.cardinality() == 87811

    print("\n", "Done !")


def test_partition():
    P = IntegerPartition([5, 3, 2, 2])
    P12 = IntegerPartitions(12)
    assert len(list(P12)) == P12.cardinality() == 77
    assert P in P12
    assert P.size == 12
    assert P.length == 4
    assert P.dictionary == {1: 0, 2: 2, 3: 1, 4: 0, 5: 1}
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
    print("\n", "Done !")


def test_random():
    print("Random uniform rooted tree", "\n")
    U = UniformRootedTree(30).get_random_element()
    U.plot(style="circular", with_levels=True, node_size=50)
    print("\n")

    UR = UniformRootedTree(5)
    R = RootedTrees(5)
    assert all(np.isclose(UR.distribution()[T.code], 1 / 9) for T in R)

    print("Random uniform recursive tree", "\n")
    T = UniformRecursiveTree(30).get_random_element()
    T.plot()
    print("\n")
    UR2 = UniformRecursiveTree(5)
    R2 = RecursiveTrees(5)
    assert all(
        np.isclose(UR2.distribution()[T.convert("code")], 1 / 24) for T in R2
    )

    print("Random uniform Plancherel tree", "\n")
    PT = PlancherelRecursiveTree(30).get_random_element()
    PT.plot()
    print("\n")

    print("Done !")


#     # TODO: RandomSubtree, RandomCut, PlancherelPartition, EwensPartition
