"""QwenVL Verifier: Vanilla Qwen2.5-VL-7B-Instruct as TTS verifier.

This serves as the ablation baseline for OmniVerifier-TTS.
In Table 3 of the paper:
  - QwenVL-TTS uses this vanilla verifier
  - OmniVerifier-TTS uses the RL-finetuned OmniVerifier-7B

Both follow the same sequential verify-then-edit loop; the only difference
is which model performs the verification.  Prompt format and parsing
are shared via the helpers in pipeline.omniverifier.
"""
from __future__ import annotations

import torch
from PIL import Image

from pipeline.omniverifier import (
    VerificationResult,
    SYS_PROMPT,
    _build_verification_question,
    _parse_verification_output,
)


class QwenVLVerifier:
    """Vanilla Qwen2.5-VL-7B-Instruct as verifier (no RL finetuning).

    Same interface as OmniVerifier so they are interchangeable in
    the SequentialTTS pipeline.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 2048,
    ):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load vanilla Qwen2.5-VL-7B-Instruct."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"[QwenVL-Verifier] Loading vanilla model from {self.model_path} ...")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        print("[QwenVL-Verifier] Model loaded successfully")

    def verify(self, image: Image.Image, prompt: str) -> VerificationResult:
        """Verify image-prompt alignment using vanilla Qwen2.5-VL."""
        raw_output = self._infer(image, prompt)
        is_aligned, explanation, edit_prompt = _parse_verification_output(raw_output)

        return VerificationResult(
            is_aligned=is_aligned,
            explanation=explanation,
            edit_instruction=edit_prompt,
            raw_output=raw_output,
        )

    def _infer(self, image: Image.Image, prompt: str) -> str:
        """Run inference using transformers.

        Uses the same prompt format as OmniVerifier: single user
        message with (question + SYS_PROMPT), no system role.
        """
        question = _build_verification_question(prompt)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question + SYS_PROMPT},
                ],
            },
        ]

        from qwen_vl_utils import process_vision_info

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

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
