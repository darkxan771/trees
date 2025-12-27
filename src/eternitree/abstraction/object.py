# Defines generic classes CombinatorialObject and CombinatorialClass

from abc import abstractmethod
from itertools import chain
from itertools import count
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterator
from typing import Protocol


class InfiniteSetError(Exception):
    pass


class CombinatorialObject(Protocol):

    def __repr__(self) -> str:
        return f"{self.category.capitalize()} with size {abs(self)}"

    def __abs__(self) -> int: ...

    def __len__(self) -> int: ...

    def __eq__(self, other) -> bool:
        return self.convert("code") == other.convert("code")

    @property
    def length(self) -> int:
        return len(self)

    @property
    def size(self) -> int:
        return abs(self)

    @property
    def category(self) -> str: ...

    @property
    def convert(self) -> Callable[[str], Any]:
        from .conversions import conversions

        return lambda S: conversions[(self.category, S)](self)


class CombinatorialClass(Collection):

    def __init__(self, category: str, order: int | None = None):
        self.category = str(category)
        self.order = order

    @classmethod
    @abstractmethod
    def generating_series(cls, N: int) -> list[int]: ...

    @classmethod
    @abstractmethod
    def iter_n(cls) -> Callable[[int], Iterator]: ...

    def __iter__(self) -> Iterator:
        if self.order is not None:
            return type(self).iter_n()(self.order)
        else:
            return chain.from_iterable(
                type(self).iter_n()(n) for n in count(1)
            ).__iter__()

    def __len__(self) -> int:
        if self.order is None:
            raise InfiniteSetError("Infinite set")
        else:
            return int(type(self).generating_series(self.order)[-1])

    def __repr__(self) -> str:
        res = f"{self.category.capitalize()}s"
        if self.order is not None:
            res += f" with size {self.order}"
        return res

    def __contains__(self, obj) -> bool:
        A = self.category == obj.category
        B = True
        if self.order is not None:
            B = self.order == obj.size
        return A and B

    @property
    def cardinality(self) -> int:
        return len(self)
