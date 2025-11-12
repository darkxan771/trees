from .recursive_trees import RecursiveTree
from .rooted_trees import RootedTree
from .containers import RecursiveTrees, RootedTrees
from .random_trees import (
    PlancherelRecursiveTree,
    UniformRecursiveTree,
    UniformRootedTree,
)
from .tests import (
    test_recursive,
    test_rooted,
    test_random,
    test_subtree_plancherel,
)


__all__ = [
    "RecursiveTree",
    "RecursiveTrees",
    "RootedTree",
    "RootedTrees",
    "PlancherelRecursiveTree",
    "UniformRecursiveTree",
    "UniformRootedTree",
    "test_recursive",
    "test_rooted",
    "test_random",
    "test_subtree_plancherel",
]


graphic_options = """
        node_size (int, 300)
        node_shape (str, "o")
        arrows (bool, True)
        arrow_size (int, 10)
        width (float, 1.0)
        node_color, edge_color
        font_size (int, 12)
        """
