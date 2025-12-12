# Defines general non-random objects:
# IntegerPartition and Tree

from .helpers import InfiniteSetError
from .partition import IntegerPartition
from .set_partition import SetPartition
from .tree import Tree

__all__ = ["InfiniteSetError", "IntegerPartition", "SetPartition", "Tree"]
