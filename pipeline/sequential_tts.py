"""Sequential OmniVerifier-TTS pipeline.

This implements the sequential test-time scaling paradigm from the paper:
1. Start with an initial generated image (step 0)
2. Use OmniVerifier to verify alignment with prompt
3. If misaligned, use the edit instruction to refine via the UMM
4. Repeat for max_rounds or until verified as aligned

This bridges image generation and editing within a unified TTS framework,
achieving iterative fine-grained optimization.

Reference: Section 4.3 and Figure 5 of the OmniVerifier paper.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from generators.base_generator import BaseGenerator
from pipeline.omniverifier import OmniVerifier, VerificationResult
from utils.io_utils import save_image, save_json
from utils.logger import get_logger


@dataclass
class TTSStepResult:
    """Result from a single TTS refinement step."""
    step: int
    image: Image.Image
    verification: VerificationResult
    edit_instruction: str = ""
    generation_time: float = 0.0
    verification_time: float = 0.0


@dataclass
class TTSResult:
    """Complete result from sequential TTS pipeline."""
    sample_id: str
    prompt: str
    final_image: Image.Image
    final_step: int                     # Which step produced the final image
    is_aligned: bool                    # Whether final image is verified aligned
    steps: list[TTSStepResult] = field(default_factory=list)
    total_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class SequentialTTS:
    """Sequential OmniVerifier-TTS pipeline.

    Implements iterative verify-then-edit refinement:
    Step 0: Initial generation
    Step 1..N: OmniVerifier verification → Edit instruction → Image editing
    """

    def __init__(
        self,
        verifier: OmniVerifier,
        generator: BaseGenerator,
        max_rounds: int = 9,
        early_stop: bool = True,
        save_intermediates: bool = True,
        output_dir: str = "results/tts",
    ):
        self.verifier = verifier
        self.generator = generator
        self.max_rounds = max_rounds
        self.early_stop = early_stop
        self.save_intermediates = save_intermediates
        self.output_dir = Path(output_dir)
        self.logger = get_logger("omniverifier.tts")

    def run(
        self,
        sample_id: str,
        prompt: str,
        initial_image: Image.Image | None = None,
    ) -> TTSResult:
        """Run the full sequential TTS pipeline for a single sample.

        Args:
            sample_id: Unique identifier for this sample.
            prompt: The text prompt.
            initial_image: Pre-generated step-0 image. If None, generates one.

        Returns:
            TTSResult with the final refined image and all intermediate steps.
        """
        start_time = time.time()
        steps: list[TTSStepResult] = []

        # Step 0: Initial generation (or use provided image)
        if initial_image is None:
            self.logger.info(f"[{sample_id}] Generating initial image (step 0)...")
            gen_start = time.time()
            result = self.generator.generate(prompt)
            current_image = result.image
            gen_time = time.time() - gen_start
        else:
            current_image = initial_image
            gen_time = 0.0

        # Verify step 0
        self.logger.info(f"[{sample_id}] Verifying step 0...")
        ver_start = time.time()
        verification = self.verifier.verify(current_image, prompt)
        ver_time = time.time() - ver_start

        step_result = TTSStepResult(
            step=0,
            image=current_image,
            verification=verification,
            generation_time=gen_time,
            verification_time=ver_time,
        )
        steps.append(step_result)

        if self.save_intermediates:
            self._save_step(sample_id, step_result)

        self.logger.info(
            f"[{sample_id}] Step 0: aligned={verification.is_aligned}"
        )

        # Early stop if already aligned
        if self.early_stop and verification.is_aligned:
            total_time = time.time() - start_time
            return TTSResult(
                sample_id=sample_id,
                prompt=prompt,
                final_image=current_image,
                final_step=0,
                is_aligned=True,
                steps=steps,
                total_time=total_time,
            )

        # Iterative refinement rounds
        for round_idx in range(1, self.max_rounds + 1):
            edit_instruction = verification.edit_instruction
            if not edit_instruction:
                self.logger.warning(
                    f"[{sample_id}] No edit instruction from verifier at step {round_idx-1}, stopping"
                )
                break

            self.logger.info(
                f"[{sample_id}] Round {round_idx}: editing with instruction: "
                f"{edit_instruction[:100]}..."
            )

            # Edit the image
            gen_start = time.time()
            try:
                edit_result = self.generator.edit(
                    image=current_image,
                    original_prompt=prompt,
                    edit_instruction=edit_instruction,
                )
                current_image = edit_result.image
            except Exception as e:
                self.logger.error(f"[{sample_id}] Edit failed at round {round_idx}: {e}")
                break
            gen_time = time.time() - gen_start

            # Verify the edited image
            ver_start = time.time()
            verification = self.verifier.verify(current_image, prompt)
            ver_time = time.time() - ver_start

            step_result = TTSStepResult(
                step=round_idx,
                image=current_image,
                verification=verification,
                edit_instruction=edit_instruction,
                generation_time=gen_time,
                verification_time=ver_time,
            )
            steps.append(step_result)

            if self.save_intermediates:
                self._save_step(sample_id, step_result)

            self.logger.info(
                f"[{sample_id}] Step {round_idx}: aligned={verification.is_aligned}"
            )

            # Early stop if aligned
            if self.early_stop and verification.is_aligned:
                break

        total_time = time.time() - start_time

        # Final result uses the last image
        final_step = steps[-1]
        result = TTSResult(
            sample_id=sample_id,
            prompt=prompt,
            final_image=final_step.image,
            final_step=final_step.step,
            is_aligned=final_step.verification.is_aligned,
            steps=steps,
            total_time=total_time,
        )

        # Save final image
        final_dir = self.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        save_image(result.final_image, final_dir / f"{sample_id}.png")

        self.logger.info(
            f"[{sample_id}] Completed: {len(steps)} steps, "
            f"aligned={result.is_aligned}, time={total_time:.1f}s"
        )

        return result

    def run_batch(
        self,
        samples: list[dict[str, Any]],
        step0_images: dict[str, Image.Image] | None = None,
    ) -> list[TTSResult]:
        """Run sequential TTS on a batch of samples.

        Args:
            samples: List of {"id": ..., "prompt": ...} dicts.
            step0_images: Optional dict mapping sample_id to pre-generated images.

        Returns:
            List of TTSResult for each sample.
        """
        results = []
        step0_images = step0_images or {}

        for i, sample in enumerate(samples):
            sample_id = sample["id"]
            prompt = sample["prompt"]
            initial_image = step0_images.get(sample_id)

            self.logger.info(f"Processing {i+1}/{len(samples)}: {sample_id}")

            try:
                result = self.run(
                    sample_id=sample_id,
                    prompt=prompt,
                    initial_image=initial_image,
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed on {sample_id}: {e}")
                continue

        return results

    def _save_step(self, sample_id: str, step: TTSStepResult) -> None:
        """Save intermediate step results."""
        step_dir = self.output_dir / "steps" / sample_id
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        save_image(step.image, step_dir / f"step_{step.step}.png")

        # Save metadata
        meta = {
            "step": step.step,
            "is_aligned": step.verification.is_aligned,
            "explanation": step.verification.explanation,
            "edit_instruction": step.edit_instruction,
            "generation_time": step.generation_time,
            "verification_time": step.verification_time,
            "raw_output": step.verification.raw_output,
        }
        save_json(meta, step_dir / f"step_{step.step}_meta.json")
