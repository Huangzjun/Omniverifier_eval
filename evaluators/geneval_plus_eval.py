"""GenEval++ evaluator using VQAScore.

The OmniVerifier paper uses VQAScore as the evaluation metric for GenEval++.
VQAScore measures image-text alignment by computing the probability that
a VQA model answers "yes" to "Does this image match the caption: {prompt}?"

Higher VQAScore indicates better alignment.

Reference:
- VQAScore: https://github.com/linzhiqiu/t2v_metrics
- GenEval++: extension of GenEval with additional compositional tasks
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from .base_evaluator import BaseEvaluator, EvalResult


class GenEvalPlusEvaluator(BaseEvaluator):
    """Evaluator for GenEval++ using VQAScore."""

    def __init__(
        self,
        model_name: str = "clip-flant5-xxl",
        device: str = "cuda",
    ):
        super().__init__(name="geneval_plus")
        self.model_name = model_name
        self.device = device
        self._scorer = None

    def load(self) -> None:
        """Load VQAScore model."""
        try:
            import t2v_metrics
            print(f"[GenEval++ Eval] Loading VQAScore model: {self.model_name} ...")
            self._scorer = t2v_metrics.VQAScore(model=self.model_name, device=self.device)
            print("[GenEval++ Eval] VQAScore model loaded")
        except ImportError:
            print("[GenEval++ Eval] WARNING: t2v_metrics not installed. Using fallback CLIP scoring.")
            print("  Install: pip install t2v-metrics")
            self._use_fallback = True

    def evaluate_single(
        self,
        image: Image.Image,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate a single image using VQAScore.

        Args:
            image: Generated image.
            prompt: Original text prompt.
            metadata: Optional metadata with category info.

        Returns:
            EvalResult with VQAScore as the score (0-100 scale).
        """
        metadata = metadata or {}
        category = metadata.get("category", metadata.get("skill", ""))

        if self._scorer is not None:
            score = self._compute_vqascore(image, prompt)
        else:
            score = self._compute_clip_fallback(image, prompt)

        # VQAScore is typically in [0, 1], convert to percentage
        score_pct = score * 100.0

        return EvalResult(
            sample_id="",
            score=score_pct,
            category=category,
            dimension=category,
            details={"vqascore_raw": score},
        )

    def _compute_vqascore(self, image: Image.Image, prompt: str) -> float:
        """Compute VQAScore for an image-prompt pair."""
        import tempfile
        import os

        # VQAScore expects file paths, so save temporarily
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            tmp_path = f.name

        try:
            score = self._scorer.score(images=[tmp_path], texts=[prompt])
            return float(score[0][0])  # Extract scalar score
        finally:
            os.unlink(tmp_path)

    def _compute_clip_fallback(self, image: Image.Image, prompt: str) -> float:
        """Fallback: compute CLIP similarity score."""
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            if not hasattr(self, "_clip_model"):
                self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
                self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

            inputs = self._clip_processor(
                text=[prompt], images=[image], return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                similarity = outputs.logits_per_image.softmax(dim=-1)
                return float(similarity[0][0].item())

        except Exception as e:
            print(f"[GenEval++ Eval] CLIP fallback failed: {e}")
            return 0.5  # Default middle score

    def evaluate_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
        sample_ids: list[str],
        metadata_list: list[dict] | None = None,
    ) -> list[EvalResult]:
        """Batch evaluate using VQAScore (more efficient)."""
        import tempfile
        import os

        metadata_list = metadata_list or [{}] * len(images)

        if self._scorer is None:
            # Fall back to sequential
            return super().evaluate_batch(images, prompts, sample_ids, metadata_list)

        # Save all images to temp files
        tmp_paths = []
        for img in images:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img.save(f, format="PNG")
                tmp_paths.append(f.name)

        try:
            # Batch VQAScore computation
            scores = self._scorer.score(images=tmp_paths, texts=prompts)

            results = []
            for i, (sid, prompt, meta) in enumerate(zip(sample_ids, prompts, metadata_list)):
                score = float(scores[i][0]) if len(scores.shape) > 1 else float(scores[i])
                category = meta.get("category", meta.get("skill", ""))

                results.append(EvalResult(
                    sample_id=sid,
                    score=score * 100.0,
                    category=category,
                    dimension=category,
                    details={"vqascore_raw": score},
                ))

            return results
        finally:
            for p in tmp_paths:
                os.unlink(p)
