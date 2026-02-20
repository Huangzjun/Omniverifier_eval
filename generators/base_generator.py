"""Abstract base class for image generators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass
class GenerationResult:
    """Result from image generation."""
    image: Image.Image
    prompt: str
    metadata: dict[str, Any] | None = None


class BaseGenerator(ABC):
    """Abstract interface for T2I generation and editing backends."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate an image from a text prompt."""
        ...

    @abstractmethod
    def edit(self, image: Image.Image, original_prompt: str, edit_instruction: str, **kwargs) -> GenerationResult:
        """Edit an existing image based on an instruction.

        Args:
            image: The input image to edit.
            original_prompt: The original generation prompt.
            edit_instruction: Natural language edit instruction from OmniVerifier.

        Returns:
            GenerationResult with the edited image.
        """
        ...
