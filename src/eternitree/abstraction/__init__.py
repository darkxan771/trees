# Defines general non-random objects:
# IntegerPartition and Tree

from .object import CombinatorialClass
from .object import CombinatorialObject
from .object import InfiniteSetError
from .partition import IntegerPartition
from .random import Random
from .set_partition import SetPartition
from .tree import Tree

__all__ = [
    "CombinatorialClass",
    "CombinatorialObject",
    "InfiniteSetError",
    "IntegerPartition",
    "Random",
    "SetPartition",
    "Tree",
]
