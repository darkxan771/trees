# TODO:  add documentation.


from collections import defaultdict
from typing import Any
from typing import Iterable
from typing import Protocol


class Random(Protocol):

    size: int | None
    object: str
    name: str
    parameter: Any = None

    def __repr__(self):
        res = f"Random {self.name.capitalize()} {self.object} with size {self.size}"
        if self.parameter is not None:
            res += f" and parameter {self.parameter}"
        return res

    @property
    def label(self) -> Any: ...

    def container(self) -> Iterable: ...

    def probability(self, obj) -> float:
        from ..random.probabilities import compute_probabilities

        if obj in self.container():
            try:
                return compute_probabilities[self.object, self.name](
                    obj, self.parameter
                )
            except KeyError:
                return self.distribution()[obj.convert("code")]
        else:
            return float(0)

    def distribution(self) -> defaultdict:
        if self.size is None:
            raise NotImplementedError
        return defaultdict(
            float,
            {
                obj.convert("code"): self.probability(obj)
                for obj in self.container()
            },
        )

    def get_random_element(self) -> Any:
        from ..random.generators import generate_random

        return generate_random[self.object, self.name](
            self.label, self.parameter
        )
