"""Qwen-Image generator backend.

Uses DashScope API to access Qwen-Image for both image generation and editing.
Qwen-Image is a Unified Multimodal Model (UMM) that supports:
- Text-to-image generation
- Image editing via natural language instructions

The OmniVerifier-TTS pipeline uses this as the base UMM for sequential refinement.

Alternatively, you can use OpenAI's GPT-Image-1 by setting the appropriate backend.
"""
from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

from PIL import Image

from .base_generator import BaseGenerator, GenerationResult


class QwenImageGenerator(BaseGenerator):
    """Qwen-Image generation and editing via DashScope API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "wanx-v1",
        image_size: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        super().__init__(name="qwen_image")
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.image_size = image_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            raise ValueError(
                "DashScope API key is required. Set DASHSCOPE_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate an image from a text prompt using Qwen-Image / DashScope.

        Uses the DashScope image synthesis API. You can also swap this
        with any other T2I model (e.g., DALL-E, Stable Diffusion).
        """
        try:
            import dashscope
            from dashscope import ImageSynthesis
        except ImportError:
            raise ImportError("dashscope package required: pip install dashscope")

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
        """Edit an image using Qwen-Image with interleaved text-image input.

        For Qwen-Image (UMM), editing is done by providing the original image
        along with the edit instruction as a multimodal prompt.

        The OmniVerifier-TTS pipeline constructs the edit prompt as:
        "Based on the original prompt: {original_prompt}. {edit_instruction}"
        """
        try:
            import dashscope
            from dashscope import MultiModalConversation
        except ImportError:
            raise ImportError("dashscope package required: pip install dashscope")

        dashscope.api_key = self.api_key

        # Encode image to base64
        image_b64 = self._image_to_base64(image)

        # Construct the multimodal editing prompt
        edit_prompt = (
            f"Please edit this image according to the following instruction. "
            f"Original description: {original_prompt}. "
            f"Edit instruction: {edit_instruction}. "
            f"Generate the edited image."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/png;base64,{image_b64}"},
                    {"text": edit_prompt},
                ],
            }
        ]

        for attempt in range(self.max_retries):
            try:
                response = MultiModalConversation.call(
                    model="qwen-vl-max",  # Use VL model for editing
                    messages=messages,
                )

                if response.status_code == 200:
                    # Extract edited image from response
                    content = response.output.choices[0].message.content
                    for item in content:
                        if "image" in item:
                            edited_image = self._download_image(item["image"])
                            return GenerationResult(
                                image=edited_image,
                                prompt=edit_prompt,
                                metadata={"edit_instruction": edit_instruction},
                            )

                print(f"[QwenImage] Edit failed (attempt {attempt+1})")
            except Exception as e:
                print(f"[QwenImage] Edit error (attempt {attempt+1}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        # Fallback: if editing fails, regenerate with combined prompt
        print("[QwenImage] Falling back to regeneration with combined prompt")
        combined_prompt = f"{original_prompt}. Additionally: {edit_instruction}"
        return self.generate(combined_prompt, **kwargs)

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

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
