"""T2I-ReasonBench evaluator.

Implements the two-stage evaluation protocol from T2I-ReasonBench:
Stage 1: Use pre-generated prompt-specific question-criteria pairs
         (already provided in the benchmark's deepseek_evaluatioin_qs/ directory)
Stage 2: Use Qwen2.5-VL as the MLLM judge to score generated images
         against the question-criteria pairs.

Each image is scored on a 0-10 scale based on how well it satisfies
the reasoning requirements of the prompt.

Reference: https://github.com/KaiyueSun98/T2I-ReasonBench
"""
from __future__ import annotations

import torch
from typing import Any

from PIL import Image

from .base_evaluator import BaseEvaluator, EvalResult


class T2IReasonBenchEvaluator(BaseEvaluator):
    """Evaluator for T2I-ReasonBench using Qwen2.5-VL as judge."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_score: float = 10.0,
    ):
        super().__init__(name="t2i_reasonbench")
        self.model_path = model_path
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.max_score = max_score
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load Qwen2.5-VL evaluation model."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"[T2I-ReasonBench Eval] Loading judge model from {self.model_path} ...")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        print("[T2I-ReasonBench Eval] Judge model loaded")

    def evaluate_single(
        self,
        image: Image.Image,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate a single image using the two-stage protocol.

        Args:
            image: Generated image.
            prompt: Original text prompt.
            metadata: Should contain 'eval_questions' with question-criteria pairs.

        Returns:
            EvalResult with score normalized to [0, 100].
        """
        metadata = metadata or {}
        eval_questions = metadata.get("eval_questions", {})
        dimension = metadata.get("dimension_full_name", "")

        if eval_questions:
            score = self._score_with_qa_pairs(image, prompt, eval_questions)
        else:
            # Fallback: direct scoring without pre-generated QA pairs
            score = self._score_direct(image, prompt)

        # Normalize to percentage
        score_pct = (score / self.max_score) * 100.0

        return EvalResult(
            sample_id="",
            score=score_pct,
            dimension=dimension,
            details={
                "raw_score": score,
                "max_score": self.max_score,
                "has_qa_pairs": bool(eval_questions),
            },
        )

    def _score_with_qa_pairs(
        self,
        image: Image.Image,
        prompt: str,
        eval_questions: dict | list,
    ) -> float:
        """Score image using pre-generated question-criteria pairs.

        The T2I-ReasonBench protocol:
        For each QA pair, ask the MLLM to evaluate if the image satisfies
        the criterion. Average the scores across all QA pairs.
        """
        # Parse QA pairs - format depends on benchmark version
        qa_pairs = self._parse_qa_pairs(eval_questions)

        if not qa_pairs:
            return self._score_direct(image, prompt)

        scores = []
        for question, criteria in qa_pairs:
            score = self._evaluate_qa_pair(image, question, criteria)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def _parse_qa_pairs(self, eval_questions: dict | list) -> list[tuple[str, str]]:
        """Parse evaluation QA pairs from various formats."""
        pairs = []

        if isinstance(eval_questions, list):
            for item in eval_questions:
                if isinstance(item, dict):
                    q = item.get("question", item.get("q", ""))
                    c = item.get("criteria", item.get("criterion", item.get("a", "")))
                    if q:
                        pairs.append((q, c))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    pairs.append((str(item[0]), str(item[1])))
        elif isinstance(eval_questions, dict):
            for q, c in eval_questions.items():
                pairs.append((str(q), str(c)))

        return pairs

    def _evaluate_qa_pair(self, image: Image.Image, question: str, criteria: str) -> float:
        """Evaluate a single QA pair against the image."""
        eval_prompt = (
            f"Look at this image and answer the following question.\n"
            f"Question: {question}\n"
        )
        if criteria:
            eval_prompt += f"The correct answer should satisfy: {criteria}\n"
        eval_prompt += (
            f"\nRate how well the image satisfies this criterion on a scale of 0-10, "
            f"where 0 means completely wrong and 10 means perfectly correct. "
            f"Output only a number."
        )

        response = self._query_model(image, eval_prompt)
        return self._extract_score(response)

    def _score_direct(self, image: Image.Image, prompt: str) -> float:
        """Direct scoring without QA pairs (fallback)."""
        eval_prompt = (
            f'Look at this image. The original prompt was: "{prompt}"\n\n'
            f"Rate how well this image matches the prompt on a scale of 0-10, "
            f"where 0 means completely mismatched and 10 means perfect match. "
            f"Consider reasoning accuracy, visual quality, and prompt adherence. "
            f"Output only a number."
        )

        response = self._query_model(image, eval_prompt)
        return self._extract_score(response)

    def _query_model(self, image: Image.Image, text_prompt: str) -> str:
        """Query the Qwen2.5-VL model with an image and text prompt."""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output_text.strip()

    def _extract_score(self, response: str) -> float:
        """Extract numerical score from model response."""
        import re

        # Try to find a number in the response
        numbers = re.findall(r"(\d+(?:\.\d+)?)", response)
        if numbers:
            score = float(numbers[0])
            return min(max(score, 0.0), self.max_score)  # Clamp to [0, max_score]

        # Heuristic fallback
        response_lower = response.lower()
        if any(w in response_lower for w in ["perfect", "excellent", "great"]):
            return 8.0
        elif any(w in response_lower for w in ["good", "mostly"]):
            return 6.0
        elif any(w in response_lower for w in ["poor", "wrong", "incorrect"]):
            return 2.0

        return 5.0  # Default middle score
