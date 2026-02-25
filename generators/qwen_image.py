"""Qwen-Image generator backend.

- Generation: DashScope API (text-to-image)
- Editing: QwenImageEditPipeline from diffusers (local diffusion model)

The OmniVerifier-TTS pipeline uses QwenImageEditPipeline("Qwen/Qwen-Image-Edit")
for iterative image editing, matching the official implementation:
https://github.com/Cominclip/OmniVerifier/blob/main/sequential_omniverifier_tts.py
"""
from __future__ import annotations

import base64
import io
import os
import random
import time
from typing import Any

import torch
from PIL import Image

from .base_generator import BaseGenerator, GenerationResult


class QwenImageGenerator(BaseGenerator):
    """Qwen-Image generation (DashScope API) and editing (QwenImageEditPipeline)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "wanx-v1",
        edit_model: str = "Qwen/Qwen-Image-Edit",
        image_size: int = 1024,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        super().__init__(name="qwen_image")
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.edit_model = edit_model
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.true_cfg_scale = true_cfg_scale
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._edit_pipe = None

    def _ensure_edit_pipeline(self) -> None:
        """Lazily load the QwenImageEditPipeline on first edit call."""
        if self._edit_pipe is not None:
            return

        from diffusers import QwenImageEditPipeline

        print(f"[QwenImage] Loading edit pipeline: {self.edit_model} ...")
        self._edit_pipe = QwenImageEditPipeline.from_pretrained(self.edit_model)
        self._edit_pipe.to(torch.bfloat16)
        self._edit_pipe.to("cuda")
        self._edit_pipe.set_progress_bar_config(disable=None)
        print("[QwenImage] Edit pipeline loaded successfully")

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate an image from a text prompt using DashScope API."""
        try:
            import dashscope
            from dashscope import ImageSynthesis
        except ImportError:
            raise ImportError("dashscope package required: pip install dashscope")

        if not self.api_key:
            raise ValueError(
                "DashScope API key is required. Set DASHSCOPE_API_KEY environment "
                "variable or pass api_key parameter."
            )

        dashscope.api_key = self.api_key

        for attempt in range(self.max_retries):
            try:
                rsp = ImageSynthesis.call(
                    model=self.model,
                    prompt=prompt,
                    n=1,
                    size=f"{self.image_size}*{self.image_size}",
                )

                if rsp.status_code == 200 and rsp.output and rsp.output.results:
                    image_url = rsp.output.results[0].url
                    image = self._download_image(image_url)
                    return GenerationResult(image=image, prompt=prompt)
                else:
                    print(f"[QwenImage] Generation failed (attempt {attempt+1}): {rsp.message}")
            except Exception as e:
                print(f"[QwenImage] Error (attempt {attempt+1}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to generate image after {self.max_retries} attempts")

    def edit(
        self,
        image: Image.Image,
        original_prompt: str,
        edit_instruction: str,
        **kwargs,
    ) -> GenerationResult:
        """Edit an image using QwenImageEditPipeline (local diffusion).

        This matches the official OmniVerifier-TTS implementation which uses
        QwenImageEditPipeline from diffusers with the edit_prompt directly
        as the prompt parameter.
        """
        self._ensure_edit_pipeline()

        result_image = self._edit_pipe(
            image=image.convert("RGB"),
            prompt=edit_instruction,
            width=self.image_size,
            height=self.image_size,
            num_inference_steps=self.num_inference_steps,
            true_cfg_scale=self.true_cfg_scale,
            generator=torch.Generator(device="cuda").manual_seed(
                random.randint(0, 10000)
            ),
        ).images[0]

        return GenerationResult(
            image=result_image,
            prompt=edit_instruction,
            metadata={"edit_instruction": edit_instruction},
        )

    @staticmethod
    def _download_image(url: str) -> Image.Image:
        """Download image from URL."""
        import requests
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")


class OpenAIImageGenerator(BaseGenerator):
    """GPT-Image-1 generation via OpenAI API (alternative backend)."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-image-1"):
        super().__init__(name="gpt_image")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.images.generate(
            model=self.model,
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        image_url = response.data[0].url
        image = self._download_image(image_url)
        return GenerationResult(image=image, prompt=prompt)

    def edit(self, image: Image.Image, original_prompt: str, edit_instruction: str, **kwargs) -> GenerationResult:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        response = client.images.edit(
            model=self.model,
            image=buffer,
            prompt=f"{original_prompt}. {edit_instruction}",
            n=1,
            size="1024x1024",
        )
        image_url = response.data[0].url
        edited_image = self._download_image(image_url)
        return GenerationResult(
            image=edited_image,
            prompt=edit_instruction,
            metadata={"edit_instruction": edit_instruction},
        )

    @staticmethod
    def _download_image(url: str) -> Image.Image:
        import requests
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
