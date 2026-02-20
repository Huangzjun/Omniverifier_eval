"""QwenVL Verifier: Vanilla Qwen2.5-VL-7B-Instruct as TTS verifier.

This serves as the ablation baseline for OmniVerifier-TTS.
In Table 3 of the paper:
  - QwenVL-TTS uses this vanilla verifier
  - OmniVerifier-TTS uses the RL-finetuned OmniVerifier-7B

Both follow the same sequential verify→edit loop, the only difference
is which model performs the verification + edit instruction extraction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from PIL import Image

from pipeline.omniverifier import VerificationResult, VERIFICATION_PROMPT


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
        max_new_tokens: int = 1024,
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
        """Verify image-prompt alignment using vanilla Qwen2.5-VL.

        Returns the same VerificationResult as OmniVerifier for
        seamless interchangeability in the TTS pipeline.
        """
        raw_output = self._infer(image, prompt)
        return self._parse_output(raw_output)

    def _infer(self, image: Image.Image, prompt: str) -> str:
        """Run inference using transformers."""
        messages = [
            {
                "role": "system",
                "content": VERIFICATION_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            f"Please check if this image strictly matches the following prompt:\n"
                            f'"{prompt}"\n\n'
                            f"Analyze the image carefully and provide your judgment."
                        ),
                    },
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

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return output_text.strip()

    def _parse_output(self, raw_output: str) -> VerificationResult:
        """Parse output — identical logic to OmniVerifier._parse_output."""
        text_lower = raw_output.lower().strip()
        is_aligned = False

        answer_patterns = [
            r"final\s*answer\s*[:：]\s*(yes|no)",
            r"my\s*(?:final\s*)?answer\s*(?:is\s*)[:：]?\s*(yes|no)",
            r"(?:the\s*)?answer\s*[:：]\s*(yes|no)",
            r"\*\*(yes|no)\*\*",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text_lower)
            if match:
                is_aligned = match.group(1) == "yes"
                break
        else:
            last_lines = text_lower.split("\n")[-3:]
            last_text = " ".join(last_lines)
            if "yes" in last_text and "no" not in last_text:
                is_aligned = True
            elif "no" in last_text:
                is_aligned = False

        edit_instruction = ""
        if not is_aligned:
            edit_instruction = self._extract_edit_instruction(raw_output)

        return VerificationResult(
            is_aligned=is_aligned,
            explanation=raw_output,
            edit_instruction=edit_instruction,
            raw_output=raw_output,
        )

    @staticmethod
    def _extract_edit_instruction(raw_output: str) -> str:
        """Extract edit instruction from verification output."""
        patterns = [
            r"edit\s*instruction\s*[:：]\s*(.+?)(?:\n|$)",
            r"should\s*be\s*(?:changed|modified|edited)\s*(?:to|by)\s*[:：]?\s*(.+?)(?:\n|$)",
            r"suggestion\s*[:：]\s*(.+?)(?:\n|$)",
            r"to\s*fix\s*this\s*[:：,]?\s*(.+?)(?:\n|$)",
            r"recommendation\s*[:：]\s*(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, raw_output, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        sentences = [s.strip() for s in raw_output.split(".") if len(s.strip()) > 20]
        if sentences:
            return ". ".join(sentences[-2:]).strip()
        return raw_output.strip()[-200:]
