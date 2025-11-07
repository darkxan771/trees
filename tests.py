from .recursive_trees import RecursiveTree, RecursiveTrees
from .rooted_trees import RootedTrees
from .random_trees import (UniformRootedTree, UniformRecursiveTree,
                           PlancherelRecursiveTree)


def test_recursive():
    T = RecursiveTree(max_size=15)
    for i in (0, 0, 0, 0, 1, 1, 0, 4, 2, 1, 6, 1, 3, 4):
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
    print(T == Rec.from_code([0, 0, 0, 0, 1, 1, 0, 4, 2, 1, 6, 1, 3, 4]))
    print(T == Rec.from_permutation([13, 12, 11, 5, 8, 7, 0,
                                     10, 6, 2, 9, 1, 3, 4]))


def test_rooted():
    T = RootedTrees(15).from_nested_list([[[[[[[[]], []], [], []],
                                         [[]], []], []]]])

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
    print(T == Root.from_code([0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 5, 4, 5, 4, 3]))


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
