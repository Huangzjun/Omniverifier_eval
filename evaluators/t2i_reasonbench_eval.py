"""T2I-ReasonBench evaluator — aligned with the official evaluation protocol.

Implements the exact two-stage evaluation from T2I-ReasonBench
(https://github.com/KaiyueSun98/T2I-ReasonBench):

Stage 1 (pre-computed): Prompt-specific question-criterion pairs generated
         by DeepSeek-R1, stored in deepseek_evaluatioin_qs/ directory.

Stage 2 (this evaluator): Multi-turn evaluation with Qwen2.5-VL as MLLM judge:
   Turn 1 — "Describe this image."
   Turn 2 — Score question-criterion pairs (0 / 0.5 / 1 per criterion)
   (Turn 3 — for entity/scientific: score other-details criteria)
   (Turn N — score image quality criteria)

Scoring per dimension:
  - Idiom / Textual:     Reasoning Accuracy = avg(reason_scores)
  - Entity / Scientific:  Reasoning Accuracy = 0.7·avg(primary) + 0.3·avg(detail)

Final scores are reported as percentages (×100), matching the paper.
"""
from __future__ import annotations

import json
import re
import torch
from typing import Any

from PIL import Image

from .base_evaluator import BaseEvaluator, EvalResult


# Dimension-specific evaluation configuration.
# Maps dimension key → eval JSON filename + which keys hold the QA pairs.
DIMENSION_EVAL_CONFIG: dict[str, dict[str, Any]] = {
    "idiom_interpretation": {
        "eval_file": "evaluation_idiom.json",
        "type": "two_score",
        "reason_key": "reason_evaluation",
        "quality_key": "quality_evaluation",
    },
    "textual_image_design": {
        "eval_file": "evaluation_textual_image.json",
        "type": "two_score",
        "reason_key": "reason_evaluation",
        "quality_key": "quality_evaluation",
    },
    "entity_reasoning": {
        "eval_file": "evaluation_entity.json",
        "type": "three_score",
        "reason_key": "entity_evaluation",
        "detail_key": "other_details_evaluation",
        "quality_key": "quality_evaluation",
        "reason_weight": 0.7,
        "detail_weight": 0.3,
    },
    "scientific_reasoning": {
        "eval_file": "evaluation_scientific.json",
        "type": "three_score",
        "reason_key": "scientific_evaluation",
        "detail_key": "other_details_evaluation",
        "quality_key": "quality_evaluation",
        "reason_weight": 0.7,
        "detail_weight": 0.3,
    },
}

_SCORE_PROMPT_TEMPLATE = """\
Based on the image and your previous description, answer the following questions: q1, q2, ...
For each question, assign a score of 1, 0.5 or 0 according to the corresponding scoring criteria: c1, c2, ...
Here are the questions and criteria: {qa_pairs}
Carefully consider the image and each question before responding, then provide your answer in json format:
{{"reason": [your detailed reasoning], "score": [s1,s2, ...]"}}"""


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from model output."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _parse_scores(raw_scores: list) -> list[float]:
    """Convert raw score values to floats, clamped to {0, 0.5, 1}."""
    parsed = []
    for s in raw_scores:
        try:
            v = float(s)
        except (ValueError, TypeError):
            v = 0.0
        v = max(0.0, min(1.0, v))
        parsed.append(v)
    return parsed


class T2IReasonBenchEvaluator(BaseEvaluator):
    """Evaluator aligned with the official T2I-ReasonBench protocol.

    Uses Qwen2.5-VL-72B-Instruct as the MLLM judge (matching the original
    benchmark), multi-turn Chain-of-Thought evaluation, and 0/0.5/1 scoring.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "auto",
        quantization: str | None = "int4",
    ):
        super().__init__(name="t2i_reasonbench")
        self.model_path = model_path
        self.device = device
        self._torch_dtype = torch_dtype
        self._quantization = quantization
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load Qwen2.5-VL judge model."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        dtype = self._torch_dtype
        if dtype != "auto":
            dtype = getattr(torch, dtype)

        quant_kwargs: dict[str, Any] = {}
        if self._quantization == "int8":
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            quant_label = " (INT8)"
        elif self._quantization == "int4":
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            quant_label = " (INT4)"
        else:
            quant_label = ""

        print(f"[T2I-ReasonBench Eval] Loading judge model: {self.model_path}{quant_label} ...")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
            **quant_kwargs,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        print("[T2I-ReasonBench Eval] Judge model loaded")

    # ─── Public interface ──────────────────────────────────────────

    def evaluate_single(
        self,
        image: Image.Image,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate a single image following the official two-stage protocol.

        Returns:
            EvalResult whose ``score`` is the Reasoning Accuracy percentage
            (0–100 scale), matching the numbers reported in the paper.
        """
        metadata = metadata or {}
        eval_questions = metadata.get("eval_questions", {})
        dim_key = metadata.get("dimension_key", "")
        dimension = metadata.get("dimension_full_name", "")

        config = DIMENSION_EVAL_CONFIG.get(dim_key)
        if not config or not eval_questions:
            accuracy, quality = self._fallback_score(image, prompt)
            return self._make_result(accuracy, quality, dimension, has_qa=False)

        # Turn 1: describe the image (Chain-of-Thought)
        description = self._describe_image(image)

        if config["type"] == "two_score":
            accuracy, quality = self._eval_two_score(
                image, description, eval_questions, config,
            )
        else:
            accuracy, quality = self._eval_three_score(
                image, description, eval_questions, config,
            )

        return self._make_result(accuracy, quality, dimension, has_qa=True,
                                 description=description)

    # ─── Dimension-specific scoring ────────────────────────────────

    def _eval_two_score(
        self,
        image: Image.Image,
        description: str,
        eval_questions: dict,
        config: dict,
    ) -> tuple[float, float]:
        """Idiom Interpretation / Textual Image Design: reason + quality."""
        reason_qs = eval_questions.get(config["reason_key"], "")
        quality_qs = eval_questions.get(config["quality_key"], "")

        reason_scores = self._score_qa_group(image, description, reason_qs)
        quality_scores = self._score_qa_group(image, description, quality_qs)

        accuracy = _avg(reason_scores)
        quality = _avg(quality_scores)
        return accuracy, quality

    def _eval_three_score(
        self,
        image: Image.Image,
        description: str,
        eval_questions: dict,
        config: dict,
    ) -> tuple[float, float]:
        """Entity / Scientific: primary_reasoning + other_details + quality."""
        primary_qs = eval_questions.get(config["reason_key"], "")
        detail_qs = eval_questions.get(config["detail_key"], "")
        quality_qs = eval_questions.get(config["quality_key"], "")

        primary_scores = self._score_qa_group(image, description, primary_qs)
        detail_scores = self._score_qa_group(image, description, detail_qs)
        quality_scores = self._score_qa_group(image, description, quality_qs)

        w1 = config.get("reason_weight", 0.7)
        w2 = config.get("detail_weight", 0.3)
        accuracy = w1 * _avg(primary_scores) + w2 * _avg(detail_scores)
        quality = _avg(quality_scores)
        return accuracy, quality

    # ─── Multi-turn model interaction ──────────────────────────────

    def _describe_image(self, image: Image.Image) -> str:
        """Turn 1: Ask the model to describe the image (CoT anchor)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        return self._generate(messages)

    def _score_qa_group(
        self,
        image: Image.Image,
        description: str,
        qa_pairs: Any,
    ) -> list[float]:
        """Score a group of question-criterion pairs in a multi-turn session.

        Builds a 3-message conversation:
          User:      [image] "Describe this image."
          Assistant:  <description>
          User:      "Based on the image ... score 0/0.5/1 ..." + QA pairs
        """
        if not qa_pairs:
            return [0.0]

        score_prompt = _SCORE_PROMPT_TEMPLATE.format(qa_pairs=qa_pairs)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {
                "role": "assistant",
                "content": description,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": score_prompt},
                ],
            },
        ]

        response = self._generate(messages)
        json_data = _extract_json(response)

        if json_data and "score" in json_data:
            return _parse_scores(json_data["score"])

        # Fallback: try to extract numbers directly
        numbers = re.findall(r"(\d+(?:\.\d+)?)", response)
        if numbers:
            return _parse_scores([float(n) for n in numbers])

        return [0.0]

    def _generate(self, messages: list[dict]) -> str:
        """Run inference on the Qwen2.5-VL model."""
        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
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
                max_new_tokens=1000,
            )

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()

    # ─── Fallback & helpers ────────────────────────────────────────

    def _fallback_score(
        self, image: Image.Image, prompt: str,
    ) -> tuple[float, float]:
        """Direct scoring when QA pairs are unavailable (not the official protocol)."""
        description = self._describe_image(image)
        score_prompt = (
            f'Based on the image and your description, the original prompt was: "{prompt}"\n\n'
            f"Rate how well this image matches the prompt on a scale of 0 to 1, "
            f"where 0 means completely wrong and 1 means perfectly correct. "
            f"Output only a number."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {"role": "assistant", "content": description},
            {
                "role": "user",
                "content": [{"type": "text", "text": score_prompt}],
            },
        ]
        response = self._generate(messages)
        numbers = re.findall(r"(\d+(?:\.\d+)?)", response)
        score = min(max(float(numbers[0]), 0.0), 1.0) if numbers else 0.5
        return score, score

    def _make_result(
        self,
        accuracy: float,
        quality: float,
        dimension: str,
        has_qa: bool,
        description: str = "",
    ) -> EvalResult:
        score_pct = accuracy * 100.0
        return EvalResult(
            sample_id="",
            score=score_pct,
            dimension=dimension,
            details={
                "reasoning_accuracy": accuracy,
                "reasoning_accuracy_pct": score_pct,
                "image_quality": quality,
                "image_quality_pct": quality * 100.0,
                "has_qa_pairs": has_qa,
                "description": description[:200] if description else "",
            },
        )


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
