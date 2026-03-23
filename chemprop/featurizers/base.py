from abc import abstractmethod
from collections.abc import Sized
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from chemprop.data.molgraph import MolGraph
else:
    MolGraph = Any

S = TypeVar("S")
T = TypeVar("T")


class Featurizer(Generic[S, T]):
    """An :class:`Featurizer` featurizes inputs type ``S`` into outputs of
    type ``T``."""

    @abstractmethod
    def __call__(self, input: S, *args, **kwargs) -> T:
        """featurize an input"""


class VectorFeaturizer(Featurizer[S, np.ndarray], Sized):
    ...


class GraphFeaturizer(Featurizer[S, MolGraph]):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        ...
