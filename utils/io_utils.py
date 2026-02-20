"""File I/O utilities."""
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


def load_yaml(path: str | Path) -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> Any:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_image(path: str | Path) -> Image.Image:
    """Load image from path."""
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: str | Path) -> None:
    """Save PIL image to path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
