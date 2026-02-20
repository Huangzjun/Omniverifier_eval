"""Abstract base dataset for benchmarks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class DataSample:
    """A single benchmark sample."""
    id: str                          # Unique sample identifier
    prompt: str                      # Text prompt for image generation
    dimension: str = ""              # Category/dimension label
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra info (eval questions, criteria, etc.)


class BaseDataset(ABC):
    """Abstract base class for benchmark datasets."""

    def __init__(self, name: str):
        self.name = name
        self._samples: list[DataSample] = []

    @abstractmethod
    def load(self) -> None:
        """Load dataset from disk."""
        ...

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> DataSample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[DataSample]:
        return iter(self._samples)

    def get_dimensions(self) -> list[str]:
        """Return unique dimensions/categories."""
        return list(set(s.dimension for s in self._samples))

    def filter_by_dimension(self, dimension: str) -> list[DataSample]:
        """Get samples for a specific dimension."""
        return [s for s in self._samples if s.dimension == dimension]
