"""OmniVerifier-7B: Generative Universal Verifier.

This module wraps the OmniVerifier-7B model (Qwen2.5-VL-7B fine-tuned via RL)
for image-prompt alignment verification.

The verification output is a JSON object:
  {"answer": true/false, "explanation": "...", "edit_prompt": "..."}

The model uses <think>...</think> tags for chain-of-thought reasoning,
and the JSON answer follows after the closing </think> tag.

Model: https://huggingface.co/comin/OmniVerifier-7B
Reference: https://github.com/Cominclip/OmniVerifier/blob/main/sequential_omniverifier_tts.py
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


@dataclass
class VerificationResult:
    """Result from OmniVerifier verification."""
    is_aligned: bool
    explanation: str
    edit_instruction: str
    raw_output: str
    confidence: float = 0.0


SYS_PROMPT = (
    " You should first think about the reasoning process in the mind "
    "and then provide the user with the answer. The reasoning process "
    "is enclosed within <think> </think> tags, i.e., "
    "<think> reasoning process here </think> answer here"
)


def _build_verification_question(prompt: str) -> str:
    """Build the verification question exactly matching the official repo."""
    return (
        f"This image was generated from the prompt: {prompt}. \n"
        " Please carefully analyze the image and determine whether all the "
        "objects, attributes, and spatial relationships mentioned in the prompt "
        "are correctly represented in the image. \n\n"
        " If the image accurately reflects the prompt, please answer 'true'; "
        "otherwise, answer 'false'. \n\n"
        " When the answer is false, you must:\n"
        " 1. Identify the main error and describe it briefly in \"explanation\".\n"
        " 2. In \"edit_prompt\", provide a **concrete image editing instruction** to fix the error. \n"
        " - The instruction must specify the exact action (e.g., add / remove / replace / move). \n"
        " - The instruction must specify the location or reference point "
        "(e.g., \"delete the bottle in the bottom-right corner\", "
        "\"add a dog next to the left pillar\"). \n"
        " - Do not give vague instructions such as \"add more bottles\" "
        "or \"ensure the count is correct\". Be precise and actionable. \n\n"
        " Respond strictly in the following JSON format: \n\n"
        " {\n"
        " \"answer\": true/false,\n"
        " \"explanation\": \"If the answer is false, briefly summarize the main error.\",\n"
        " \"edit_prompt\": \"If the answer is false, provide a concrete and "
        "location-specific editing instruction.\"\n"
        " }\n"
    )


def _parse_verification_output(raw_output: str) -> tuple[bool, str, str]:
    """Parse the verification output (with <think> tags and JSON).

    Handles multiple output formats:
    1. <think>...</think> { JSON }          (OmniVerifier-7B style)
    2. ```json\n{ JSON }\n```              (markdown code block)
    3. Raw JSON string                      (plain JSON)
    4. JSON embedded in natural language     (regex extraction)

    Returns (is_aligned, explanation, edit_prompt).
    """
    try:
        # Strip <think> block if present
        text = raw_output
        if "</think>" in text:
            text = text.split("</think>")[1].strip()

        # Try direct JSON parse first
        try:
            output_json = json.loads(text)
        except json.JSONDecodeError:
            # Strip markdown code block markers: ```json ... ``` or ``` ... ```
            stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
            try:
                output_json = json.loads(stripped)
            except json.JSONDecodeError:
                # Last resort: find first { ... } block via regex
                match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
                if match:
                    output_json = json.loads(match.group())
                else:
                    return False, "", "remain unchanged"

        answer = output_json.get("answer", False)
        if isinstance(answer, str):
            answer = answer.lower().strip() == "true"
        explanation = output_json.get("explanation", "")
        edit_prompt = output_json.get("edit_prompt", "remain unchanged")
        return bool(answer), explanation, edit_prompt
    except (json.JSONDecodeError, IndexError, TypeError):
        return False, "", "remain unchanged"


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
        max_new_tokens: int = 2048,
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
        """Verify whether an image is aligned with a text prompt."""
        if self.use_vllm:
            raw_output = self._infer_vllm(image, prompt)
        else:
            raw_output = self._infer_transformers(image, prompt)

        is_aligned, explanation, edit_prompt = _parse_verification_output(raw_output)

        return VerificationResult(
            is_aligned=is_aligned,
            explanation=explanation,
            edit_instruction=edit_prompt,
            raw_output=raw_output,
        )

    def _infer_transformers(self, image: Image.Image, prompt: str) -> str:
        """Run inference using transformers.

        Matches the official repo: single user message with
        (question + SYS_PROMPT), no system role.
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

    def _infer_vllm(self, image: Image.Image, prompt: str) -> str:
        """Run inference using vLLM."""
        question = _build_verification_question(prompt)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._pil_to_data_url(image)}},
                    {"type": "text", "text": question + SYS_PROMPT},
                ],
            },
        ]

        outputs = self._model.chat(messages, sampling_params=self._sampling_params)
        return outputs[0].outputs[0].text

    @staticmethod
    def _pil_to_data_url(image: Image.Image) -> str:
        """Convert PIL image to data URL for vLLM."""
        import base64
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
