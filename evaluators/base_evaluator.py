"""Abstract base class for benchmark evaluators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class EvalResult:
    """Evaluation result for a single sample."""
    sample_id: str
    score: float
    category: str = ""
    dimension: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(ABC):
    """Abstract interface for benchmark evaluators."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load(self) -> None:
        """Load evaluation model/resources."""
        ...

    @abstractmethod
    def evaluate_single(self, image: Image.Image, prompt: str, metadata: dict[str, Any] | None = None) -> EvalResult:
        """Evaluate a single image against its prompt."""
        ...

    def evaluate_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
        sample_ids: list[str],
        metadata_list: list[dict] | None = None,
    ) -> list[EvalResult]:
        """Evaluate a batch of images. Default: sequential."""
        metadata_list = metadata_list or [{}] * len(images)
        results = []
        for img, prompt, sid, meta in zip(images, prompts, sample_ids, metadata_list):
            result = self.evaluate_single(img, prompt, meta)
            result.sample_id = sid
            results.append(result)
        return results
