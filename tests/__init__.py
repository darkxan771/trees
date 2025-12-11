# Some tests to make sure that the
# modifications are not breaking the code


from .tests import test_partition, test_random, test_recursive, test_rooted

__all__ = ["test_recursive", "test_rooted", "test_partition", "test_random"]
