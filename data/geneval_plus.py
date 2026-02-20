"""GenEval++ dataset loader.

GenEval++ extends GenEval with additional compositional generation tasks.
It evaluates object co-occurrence, position, count, colors, and attribute binding.

The OmniVerifier paper uses VQAScore as the evaluation metric for GenEval++.
We support loading prompts from both the original GenEval format and
custom GenEval++ JSON files.

References:
- GenEval: https://github.com/djghosh13/geneval
- VQAScore: https://github.com/linzhiqiu/t2v_metrics
"""
from __future__ import annotations

import json
from pathlib import Path

from .base_dataset import BaseDataset, DataSample


# GenEval original categories
GENEVAL_CATEGORIES = [
    "single_object",
    "two_objects",
    "counting",
    "colors",
    "position",
    "color_attribution",
]


class GenEvalPlusDataset(BaseDataset):
    """Loader for GenEval++ benchmark."""

    def __init__(
        self,
        prompts_file: str = "data/geneval_plus_prompts.json",
        geneval_metadata_file: str | None = None,
    ):
        super().__init__(name="geneval_plus")
        self.prompts_file = Path(prompts_file)
        self.geneval_metadata_file = geneval_metadata_file

    def load(self) -> None:
        """Load GenEval++ prompts.

        Supports two formats:
        1. JSON list of {"prompt": ..., "category": ..., ...}
        2. JSONL from original GenEval (one JSON object per line)
        """
        self._samples = []

        if not self.prompts_file.exists():
            print(f"[GenEval++] Prompt file not found: {self.prompts_file}")
            print("[GenEval++] Attempting to load from original GenEval metadata...")
            self._load_from_geneval_original()
            return

        suffix = self.prompts_file.suffix.lower()

        if suffix == ".jsonl":
            with open(self.prompts_file, "r") as f:
                data = [json.loads(line) for line in f if line.strip()]
        else:
            with open(self.prompts_file, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Handle {"prompts": [...]} format
                    data = data.get("prompts", data.get("data", []))

        for idx, entry in enumerate(data):
            if isinstance(entry, str):
                prompt_text = entry
                category = "unknown"
            elif isinstance(entry, dict):
                prompt_text = entry.get("prompt", entry.get("text", ""))
                category = entry.get("category", entry.get("skill", "unknown"))
            else:
                continue

            self._samples.append(
                DataSample(
                    id=f"geneval_{idx:04d}",
                    prompt=prompt_text,
                    dimension=category,
                    metadata=entry if isinstance(entry, dict) else {},
                )
            )

        print(f"[GenEval++] Loaded {len(self._samples)} prompts")

    def _load_from_geneval_original(self) -> None:
        """Fallback: load from original GenEval metadata.jsonl."""
        meta_path = self.geneval_metadata_file or "data/geneval/metadata.jsonl"
        meta_path = Path(meta_path)

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Neither GenEval++ prompts nor GenEval metadata found.\n"
                f"Please either:\n"
                f"  1. Place GenEval++ prompts at {self.prompts_file}\n"
                f"  2. Clone GenEval: git clone https://github.com/djghosh13/geneval.git data/geneval\n"
            )

        with open(meta_path, "r") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                entry = json.loads(line)
                self._samples.append(
                    DataSample(
                        id=f"geneval_{idx:04d}",
                        prompt=entry.get("prompt", ""),
                        dimension=entry.get("skill", "unknown"),
                        metadata=entry,
                    )
                )

        print(f"[GenEval++] Loaded {len(self._samples)} prompts from GenEval metadata")

    @staticmethod
    def create_prompts_file(output_path: str = "data/geneval_plus_prompts.json") -> None:
        """Helper to create a GenEval++ prompts file from GenEval metadata."""
        print("Use the original GenEval repo to generate the prompts file.")
        print("git clone https://github.com/djghosh13/geneval.git")
        print("Then convert metadata.jsonl to the required format.")
