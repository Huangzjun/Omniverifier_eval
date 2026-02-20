"""Parallel TTS: Best-of-N baseline.

Generates N images in parallel and uses OmniVerifier to select the best one.
This serves as the baseline comparison for sequential TTS in Table 3/4.

The parallel approach generates multiple candidates and picks the one
verified as aligned (or the "best" one by some heuristic).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from generators.base_generator import BaseGenerator
from pipeline.omniverifier import OmniVerifier, VerificationResult
from utils.logger import get_logger


@dataclass
class ParallelTTSResult:
    """Result from parallel TTS (Best-of-N)."""
    sample_id: str
    prompt: str
    best_image: Image.Image
    best_index: int
    is_aligned: bool
    all_verifications: list[VerificationResult] = field(default_factory=list)
    total_time: float = 0.0


class ParallelTTS:
    """Parallel Test-Time Scaling (Best-of-N) baseline.

    Generate N images, verify each with OmniVerifier, select the best.
    """

    def __init__(
        self,
        verifier: OmniVerifier,
        generator: BaseGenerator,
        n_candidates: int = 4,
    ):
        self.verifier = verifier
        self.generator = generator
        self.n_candidates = n_candidates
        self.logger = get_logger("omniverifier.parallel_tts")

    def run(self, sample_id: str, prompt: str) -> ParallelTTSResult:
        """Generate N candidates and select the best."""
        start_time = time.time()

        # Generate N candidate images
        candidates: list[Image.Image] = []
        for i in range(self.n_candidates):
            self.logger.info(f"[{sample_id}] Generating candidate {i+1}/{self.n_candidates}")
            result = self.generator.generate(prompt)
            candidates.append(result.image)

        # Verify each candidate
        verifications: list[VerificationResult] = []
        for i, img in enumerate(candidates):
            self.logger.info(f"[{sample_id}] Verifying candidate {i+1}/{self.n_candidates}")
            ver = self.verifier.verify(img, prompt)
            verifications.append(ver)

        # Select best: prefer aligned images, otherwise pick first
        best_idx = 0
        for i, ver in enumerate(verifications):
            if ver.is_aligned:
                best_idx = i
                break

        total_time = time.time() - start_time

        return ParallelTTSResult(
            sample_id=sample_id,
            prompt=prompt,
            best_image=candidates[best_idx],
            best_index=best_idx,
            is_aligned=verifications[best_idx].is_aligned,
            all_verifications=verifications,
            total_time=total_time,
        )
