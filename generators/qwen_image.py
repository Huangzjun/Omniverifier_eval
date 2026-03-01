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
        self._edit_pipe = QwenImageEditPipeline.from_pretrained(
            self.edit_model,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self._edit_pipe.enable_attention_slicing()
        self._edit_pipe.enable_vae_tiling()
        self._edit_pipe.set_progress_bar_config(disable=None)
        print("[QwenImage] Edit pipeline loaded (on GPU, attention slicing + VAE tiling)")

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
        torch.cuda.empty_cache()

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

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-image-1",
        max_retries: int = 5,
        retry_delay: float = 3.0,
        timeout: float = 120.0,
    ):
        super().__init__(name="gpt_image")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    def _get_client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, timeout=self.timeout)

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        client = self._get_client()

        for attempt in range(self.max_retries):
            try:
                response = client.images.generate(
                    model=self.model,
                    prompt=prompt,
                    n=1,
                    size="1024x1024",
                )
                image = self._extract_image(response.data[0])
                return GenerationResult(image=image, prompt=prompt)
            except Exception as e:
                self._log_openai_error("Generate", attempt, prompt, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Failed to generate image after {self.max_retries} attempts")

    def edit(self, image: Image.Image, original_prompt: str, edit_instruction: str, **kwargs) -> GenerationResult:
        client = self._get_client()
        combined_prompt = f"{original_prompt}. {edit_instruction}"
        if len(combined_prompt) > 1000:
            combined_prompt = combined_prompt[:997] + "..."

        rgba_image = image.convert("RGBA").resize((1024, 1024), Image.LANCZOS)

        mask = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))

        for attempt in range(self.max_retries):
            try:
                img_buf = io.BytesIO()
                rgba_image.save(img_buf, format="PNG")
                img_buf.seek(0)
                img_buf.name = "image.png"

                mask_buf = io.BytesIO()
                mask.save(mask_buf, format="PNG")
                mask_buf.seek(0)
                mask_buf.name = "mask.png"

                response = client.images.edit(
                    model="dall-e-2",
                    image=img_buf,
                    mask=mask_buf,
                    prompt=combined_prompt,
                    n=1,
                    size="1024x1024",
                )
                edited_image = self._extract_image(response.data[0])
                return GenerationResult(
                    image=edited_image,
                    prompt=edit_instruction,
                    metadata={"edit_instruction": edit_instruction},
                )
            except Exception as e:
                self._log_openai_error("Edit", attempt, combined_prompt, e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Failed to edit image after {self.max_retries} attempts")

    def _log_openai_error(self, action: str, attempt: int, prompt: str, exc: Exception) -> None:
        """Print structured OpenAI API error details to stderr."""
        import sys
        prompt_preview = prompt[:200] + ("..." if len(prompt) > 200 else "")

        try:
            from openai import APIStatusError
            if isinstance(exc, APIStatusError):
                body = getattr(exc, "body", None) or {}
                err = body.get("error", {}) if isinstance(body, dict) else {}
                req_id = "N/A"
                if exc.response is not None:
                    req_id = exc.response.headers.get("x-request-id", "N/A")
                print(
                    f"[GPT-Image] {action} failed (attempt {attempt+1}/{self.max_retries})\n"
                    f"  status_code  : {exc.status_code}\n"
                    f"  error.type   : {err.get('type', 'N/A')}\n"
                    f"  error.code   : {err.get('code', 'N/A')}\n"
                    f"  error.param  : {err.get('param', 'N/A')}\n"
                    f"  error.message: {err.get('message', str(exc))}\n"
                    f"  request_id   : {req_id}\n"
                    f"  prompt_preview: {prompt_preview}",
                    file=sys.stderr, flush=True,
                )
                return
        except ImportError:
            pass

        print(
            f"[GPT-Image] {action} failed (attempt {attempt+1}/{self.max_retries})\n"
            f"  exception_type: {type(exc).__name__}\n"
            f"  message       : {exc}\n"
            f"  prompt_preview: {prompt_preview}",
            file=sys.stderr, flush=True,
        )

    @staticmethod
    def _extract_image(data) -> Image.Image:
        """Extract image from API response (supports both URL and base64)."""
        if data.b64_json:
            import base64
            img_bytes = base64.b64decode(data.b64_json)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif data.url:
            import requests
            resp = requests.get(data.url, timeout=60)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            raise ValueError("API response contains neither b64_json nor url")
