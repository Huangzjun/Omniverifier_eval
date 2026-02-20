"""OmniVerifier-7B: Generative Universal Verifier.

This module wraps the OmniVerifier-7B model (Qwen2.5-VL-7B fine-tuned via RL)
for image-prompt alignment verification. The model outputs:
1. A binary judgment: "yes" (aligned) or "no" (misaligned)
2. An explanation of misalignment + edit instruction

The OmniVerifier serves as the "misalignment-finder" in the TTS pipeline.

Model: https://huggingface.co/comin/OmniVerifier-7B
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


@dataclass
class VerificationResult:
    """Result from OmniVerifier verification."""
    is_aligned: bool            # Whether the image is aligned with the prompt
    explanation: str             # Explanation of the verification result
    edit_instruction: str        # If misaligned, suggested edit instruction
    raw_output: str              # Raw model output
    confidence: float = 0.0     # Optional confidence score


# The system prompt used by OmniVerifier for verification
VERIFICATION_PROMPT = (
    "You are a professional image evaluator. Given an image and a text prompt, "
    "you need to determine whether the image is strictly aligned with the prompt. "
    "Please first provide your analysis, then give your final answer as 'yes' or 'no'. "
    "If the answer is 'no', please also provide a specific edit instruction "
    "describing what should be changed to make the image aligned with the prompt."
)


class OmniVerifier:
    """OmniVerifier-7B model for image-prompt alignment verification.

    Supports two inference backends:
    1. Transformers (default): Direct HuggingFace inference
    2. vLLM: Faster inference with vLLM serving
    """

    def __init__(
        self,
        model_path: str = "comin/OmniVerifier-7B",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
        use_vllm: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.max_new_tokens = max_new_tokens
        self.use_vllm = use_vllm

        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load the model and processor."""
        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_transformers(self) -> None:
        """Load model using transformers."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"[OmniVerifier] Loading model from {self.model_path} ...")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        print("[OmniVerifier] Model loaded successfully")

    def _load_vllm(self) -> None:
        """Load model using vLLM for faster inference."""
        from vllm import LLM, SamplingParams

        print(f"[OmniVerifier] Loading model with vLLM from {self.model_path} ...")
        self._model = LLM(
            model=self.model_path,
            dtype=str(self.torch_dtype).split(".")[-1],
            trust_remote_code=True,
            max_model_len=8192,
        )
        self._sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=0.1,
            top_p=0.95,
        )
        print("[OmniVerifier] vLLM model loaded successfully")

    def verify(self, image: Image.Image, prompt: str) -> VerificationResult:
        """Verify whether an image is aligned with a text prompt.

        Args:
            image: The generated image to verify.
            prompt: The original text prompt.

        Returns:
            VerificationResult with alignment judgment and edit instruction.
        """
        if self.use_vllm:
            raw_output = self._infer_vllm(image, prompt)
        else:
            raw_output = self._infer_transformers(image, prompt)

        return self._parse_output(raw_output)

    def _infer_transformers(self, image: Image.Image, prompt: str) -> str:
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

        # Prepare inputs
        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

        # Decode only new tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output_text.strip()

    def _infer_vllm(self, image: Image.Image, prompt: str) -> str:
        """Run inference using vLLM."""
        from vllm import SamplingParams

        user_text = (
            f"Please check if this image strictly matches the following prompt:\n"
            f'"{prompt}"\n\n'
            f"Analyze the image carefully and provide your judgment."
        )

        messages = [
            {"role": "system", "content": VERIFICATION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._pil_to_data_url(image)}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        outputs = self._model.chat(messages, sampling_params=self._sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _parse_output(self, raw_output: str) -> VerificationResult:
        """Parse the model output into structured VerificationResult.

        The model typically outputs analysis followed by a conclusion containing
        'yes' or 'no', and if 'no', an edit instruction.
        """
        text_lower = raw_output.lower().strip()

        # Determine alignment - look for final answer
        # Common patterns: "Final answer: yes/no", "my answer is yes/no",
        # or simply ending with "yes"/"no"
        is_aligned = False

        # Check for explicit answer patterns
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
            # Fallback: check if the last line contains yes/no
            last_lines = text_lower.split("\n")[-3:]
            last_text = " ".join(last_lines)
            if "yes" in last_text and "no" not in last_text:
                is_aligned = True
            elif "no" in last_text:
                is_aligned = False

        # Extract edit instruction if misaligned
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
        """Extract the edit instruction from the verification output."""
        # Look for common edit instruction patterns
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

        # Fallback: use the last substantial sentence as the edit instruction
        sentences = [s.strip() for s in raw_output.split(".") if len(s.strip()) > 20]
        if sentences:
            # Return the last few sentences as they often contain the correction
            return ". ".join(sentences[-2:]).strip()

        return raw_output.strip()[-200:]  # Last 200 chars as fallback

    @staticmethod
    def _pil_to_data_url(image: Image.Image) -> str:
        """Convert PIL image to data URL for vLLM."""
        import base64
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
