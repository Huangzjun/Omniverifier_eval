"""Diffusion model generators for baseline evaluation.

These are pure T2I models (no native editing capability).
They can only be used for step-0 generation as baselines in Table 3,
NOT for the OmniVerifier-TTS verify→edit loop.

Supported models:
- Stable Diffusion 3 / 3.5 (SD3-Medium, SD3.5-Large, etc.)
- FLUX.1 (dev, schnell)
- SDXL
- HiDream, Playground, PixArt, etc.
"""
from __future__ import annotations

import torch
from PIL import Image

from .base_generator import BaseGenerator, GenerationResult


class DiffusionGenerator(BaseGenerator):
    """Generic diffusion model generator using diffusers pipeline.

    Supports SD3, SDXL, FLUX, PixArt, and other diffusers-compatible models.
    """

    # Preset configurations for common models
    PRESETS = {
        "sd3-medium": {
            "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
            "pipeline_class": "StableDiffusion3Pipeline",
        },
        "sd3.5-medium": {
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
            "pipeline_class": "StableDiffusion3Pipeline",
        },
        "sd3.5-large": {
            "model_id": "stabilityai/stable-diffusion-3.5-large",
            "pipeline_class": "StableDiffusion3Pipeline",
        },
        "flux-dev": {
            "model_id": "black-forest-labs/FLUX.1-dev",
            "pipeline_class": "FluxPipeline",
        },
        "flux-schnell": {
            "model_id": "black-forest-labs/FLUX.1-schnell",
            "pipeline_class": "FluxPipeline",
        },
        "sdxl": {
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "pipeline_class": "StableDiffusionXLPipeline",
        },
        "playground-v2.5": {
            "model_id": "playgroundai/playground-v2.5-1024px-aesthetic",
            "pipeline_class": "StableDiffusionXLPipeline",
        },
        "pixart-sigma": {
            "model_id": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            "pipeline_class": "PixArtSigmaPipeline",
        },
        "hidream-i1": {
            "model_id": "HiDream-ai/HiDream-I1-Full",
            "pipeline_class": "HiDreamImagePipeline",
        },
    }

    def __init__(
        self,
        model_name: str = "flux-dev",
        model_id: str | None = None,
        pipeline_class: str | None = None,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        height: int = 1024,
        width: int = 1024,
    ):
        super().__init__(name=f"diffusion_{model_name}")
        self.model_name = model_name
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.height = height
        self.width = width

        # Resolve preset or custom config
        if model_name in self.PRESETS:
            preset = self.PRESETS[model_name]
            self.model_id = model_id or preset["model_id"]
            self.pipeline_class_name = pipeline_class or preset["pipeline_class"]
        else:
            if model_id is None:
                raise ValueError(
                    f"Unknown preset '{model_name}'. Available: {list(self.PRESETS.keys())}. "
                    f"Or provide model_id explicitly."
                )
            self.model_id = model_id
            self.pipeline_class_name = pipeline_class or "StableDiffusionPipeline"

        # Default inference params per model family
        self._num_inference_steps = num_inference_steps
        self._guidance_scale = guidance_scale
        self._pipe = None

    def load(self) -> None:
        """Load the diffusion pipeline."""
        import diffusers

        pipe_class = getattr(diffusers, self.pipeline_class_name)
        print(f"[DiffusionGen] Loading {self.model_name} ({self.model_id})...")

        self._pipe = pipe_class.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
        ).to(self.device)

        # Enable memory optimizations
        if hasattr(self._pipe, "enable_model_cpu_offload"):
            try:
                self._pipe.enable_model_cpu_offload()
            except Exception:
                pass

        print(f"[DiffusionGen] {self.model_name} loaded successfully")

    @property
    def num_inference_steps(self) -> int:
        if self._num_inference_steps is not None:
            return self._num_inference_steps
        # Defaults per model family
        defaults = {
            "flux-schnell": 4,
            "flux-dev": 28,
            "sd3-medium": 28,
            "sd3.5-medium": 28,
            "sd3.5-large": 28,
            "sdxl": 30,
            "playground-v2.5": 50,
            "pixart-sigma": 20,
        }
        return defaults.get(self.model_name, 28)

    @property
    def guidance_scale(self) -> float:
        if self._guidance_scale is not None:
            return self._guidance_scale
        defaults = {
            "flux-schnell": 0.0,
            "flux-dev": 3.5,
            "sd3-medium": 7.0,
            "sd3.5-medium": 5.0,
            "sd3.5-large": 4.5,
            "sdxl": 7.5,
            "playground-v2.5": 3.0,
            "pixart-sigma": 4.5,
        }
        return defaults.get(self.model_name, 7.0)

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate an image from a text prompt."""
        if self._pipe is None:
            self.load()

        gen_kwargs = {
            "prompt": prompt,
            "num_inference_steps": kwargs.get("num_inference_steps", self.num_inference_steps),
            "guidance_scale": kwargs.get("guidance_scale", self.guidance_scale),
            "height": kwargs.get("height", self.height),
            "width": kwargs.get("width", self.width),
        }

        # Some pipelines don't support height/width (Flux)
        try:
            output = self._pipe(**gen_kwargs)
        except TypeError:
            gen_kwargs.pop("height", None)
            gen_kwargs.pop("width", None)
            output = self._pipe(**gen_kwargs)

        image = output.images[0]
        return GenerationResult(image=image, prompt=prompt)

    def generate_batch(self, prompts: list[str], **kwargs) -> list[GenerationResult]:
        """Generate images for multiple prompts in a single forward pass."""
        if self._pipe is None:
            self.load()

        gen_kwargs = {
            "prompt": prompts,
            "num_inference_steps": kwargs.get("num_inference_steps", self.num_inference_steps),
            "guidance_scale": kwargs.get("guidance_scale", self.guidance_scale),
            "height": kwargs.get("height", self.height),
            "width": kwargs.get("width", self.width),
        }

        try:
            output = self._pipe(**gen_kwargs)
        except TypeError:
            gen_kwargs.pop("height", None)
            gen_kwargs.pop("width", None)
            output = self._pipe(**gen_kwargs)

        return [
            GenerationResult(image=img, prompt=p)
            for img, p in zip(output.images, prompts)
        ]

    def edit(self, image: Image.Image, original_prompt: str, edit_instruction: str, **kwargs) -> GenerationResult:
        """Pure T2I models cannot natively edit. Regenerate with combined prompt."""
        combined = f"{original_prompt}. {edit_instruction}"
        print(f"[DiffusionGen] WARNING: {self.model_name} has no native editing. Regenerating with combined prompt.")
        return self.generate(combined, **kwargs)


class AutoregressiveGenerator(BaseGenerator):
    """Autoregressive T2I models (Show-O, Janus-Pro, Emu3, etc.)

    These typically also lack native editing capability,
    but are included as baselines for the benchmarks.
    """

    PRESETS = {
        "janus-pro-7b": {
            "model_id": "deepseek-ai/Janus-Pro-7B",
        },
        "show-o": {
            "model_id": "showlab/show-o",
        },
        "emu3": {
            "model_id": "BAAI/Emu3-Gen",
        },
    }

    def __init__(self, model_name: str = "janus-pro-7b", model_id: str | None = None, device: str = "cuda"):
        super().__init__(name=f"ar_{model_name}")
        self.model_name = model_name
        self.device = device

        if model_name in self.PRESETS:
            self.model_id = model_id or self.PRESETS[model_name]["model_id"]
        elif model_id:
            self.model_id = model_id
        else:
            raise ValueError(f"Unknown AR model: {model_name}")

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate image. Implementation varies per model — override for specific models."""
        raise NotImplementedError(
            f"AR model {self.model_name} requires a model-specific generation implementation. "
            f"Please subclass AutoregressiveGenerator and implement generate()."
        )

    def edit(self, image: Image.Image, original_prompt: str, edit_instruction: str, **kwargs) -> GenerationResult:
        combined = f"{original_prompt}. {edit_instruction}"
        return self.generate(combined, **kwargs)
