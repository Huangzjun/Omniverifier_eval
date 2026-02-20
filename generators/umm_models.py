"""Generators for Janus-Pro and BAGEL unified multimodal models.

Both are autoregressive models that support text-to-image generation.
Used as step-0 baselines only (no native editing for TTS loop).
"""
from __future__ import annotations

import torch
from PIL import Image

from .base_generator import BaseGenerator, GenerationResult


class JanusProGenerator(BaseGenerator):
    """Janus-Pro-7B text-to-image generation.

    Janus-Pro is a unified multimodal model from DeepSeek
    that uses separate visual encoding paths for understanding
    and generation tasks.

    Reference: https://github.com/deepseek-ai/Janus
    """

    def __init__(
        self,
        model_path: str = "deepseek-ai/Janus-Pro-7B",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ):
        super().__init__(name="janus_pro")
        self.model_path = model_path
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self._model = None
        self._processor = None
        self._tokenizer = None

    def load(self) -> None:
        """Load Janus-Pro model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[JanusPro] Loading model from {self.model_path} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

        # Load VL processor if available
        try:
            from janus.models import VLChatProcessor
            self._processor = VLChatProcessor.from_pretrained(self.model_path)
        except ImportError:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        print("[JanusPro] Model loaded successfully")

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate image using Janus-Pro."""
        if self._model is None:
            self.load()

        # Janus-Pro generation uses a specific conversation format
        conversation = [{"role": "User", "content": prompt}, {"role": "Assistant", "content": ""}]

        # Try the official Janus generation API
        try:
            return self._generate_official(prompt, conversation, **kwargs)
        except Exception:
            return self._generate_transformers(prompt, **kwargs)

    def _generate_official(self, prompt: str, conversation: list, **kwargs) -> GenerationResult:
        """Generate using Janus official API."""
        from janus.models import MultiModalityCausalLM
        from janus.utils.io import generate_image

        images = generate_image(
            model=self._model,
            processor=self._processor,
            prompt=prompt,
            num_images=1,
            cfg_weight=kwargs.get("cfg_weight", 5.0),
            temperature=kwargs.get("temperature", 1.0),
            image_token_num_per_image=kwargs.get("image_token_num", 576),
            patch_size=16,
            image_size=384,
        )
        image = images[0] if isinstance(images, list) else images
        if not isinstance(image, Image.Image):
            import numpy as np
            image = Image.fromarray(np.uint8(image))
        return GenerationResult(image=image.convert("RGB"), prompt=prompt)

    def _generate_transformers(self, prompt: str, **kwargs) -> GenerationResult:
        """Fallback generation via transformers generate()."""
        # Construct the generation input
        input_text = f"<image_generation>{prompt}"
        inputs = self._tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 4096),
                do_sample=True,
                temperature=kwargs.get("temperature", 1.0),
            )

        # Decode image tokens from output
        generated_ids = outputs[0, inputs.input_ids.shape[1]:]
        image = self._decode_image_tokens(generated_ids)
        return GenerationResult(image=image, prompt=prompt)

    def _decode_image_tokens(self, token_ids: torch.Tensor) -> Image.Image:
        """Decode generated tokens to image using the model's visual decoder."""
        if hasattr(self._model, "decode_image"):
            image = self._model.decode_image(token_ids.unsqueeze(0))
            if isinstance(image, torch.Tensor):
                import numpy as np
                image = image.squeeze().permute(1, 2, 0).cpu().numpy()
                image = (image * 255).clip(0, 255).astype(np.uint8)
                return Image.fromarray(image)
            return image
        raise RuntimeError("Cannot decode image tokens - model lacks decode_image method")

    def edit(self, image: Image.Image, original_prompt: str, edit_instruction: str, **kwargs) -> GenerationResult:
        """Janus-Pro: no native editing, regenerate with combined prompt."""
        combined = f"{original_prompt}. {edit_instruction}"
        return self.generate(combined, **kwargs)


class BAGELGenerator(BaseGenerator):
    """BAGEL text-to-image generation.

    BAGEL is a unified multimodal model from ByteDance Seed that
    supports both understanding and generation with optional
    "thinking" mode for improved reasoning.

    Reference: https://github.com/bytedance-seed/BAGEL
    """

    def __init__(
        self,
        model_path: str = "ByteDance-Seed/BAGEL-7B-MoT",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        use_thinking: bool = True,
    ):
        super().__init__(name="bagel")
        self.model_path = model_path
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.use_thinking = use_thinking
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load BAGEL model."""
        print(f"[BAGEL] Loading model from {self.model_path} ...")

        try:
            # Try official BAGEL loading
            from bagel import BAGELModel
            self._model = BAGELModel.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
            ).to(self.device).eval()
        except ImportError:
            # Fallback to transformers
            from transformers import AutoModelForCausalLM, AutoProcessor
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
            ).to(self.device).eval()
            self._processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )

        print(f"[BAGEL] Model loaded (thinking={'on' if self.use_thinking else 'off'})")

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate image using BAGEL."""
        if self._model is None:
            self.load()

        # BAGEL supports a "thinking" mode that improves generation quality
        gen_prompt = prompt
        if self.use_thinking:
            gen_prompt = f"<think>\n</think>\n{prompt}"

        try:
            return self._generate_official(gen_prompt, **kwargs)
        except Exception:
            return self._generate_transformers(gen_prompt, **kwargs)

    def _generate_official(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate using BAGEL official API."""
        image = self._model.generate_image(
            prompt=prompt,
            num_inference_steps=kwargs.get("num_inference_steps", 50),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            height=kwargs.get("height", 1024),
            width=kwargs.get("width", 1024),
        )
        if isinstance(image, list):
            image = image[0]
        if not isinstance(image, Image.Image):
            import numpy as np
            image = Image.fromarray(np.uint8(image))
        return GenerationResult(image=image.convert("RGB"), prompt=prompt)

    def _generate_transformers(self, prompt: str, **kwargs) -> GenerationResult:
        """Fallback generation via transformers."""
        if self._processor is None:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )

        inputs = self._processor(text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 8192),
                do_sample=True,
                temperature=kwargs.get("temperature", 1.0),
            )

        # Extract image from outputs
        generated_ids = outputs[0, inputs.input_ids.shape[1]:]
        if hasattr(self._model, "decode_image"):
            image = self._model.decode_image(generated_ids.unsqueeze(0))
        else:
            raise RuntimeError("Cannot decode - model lacks decode_image method")

        if isinstance(image, torch.Tensor):
            import numpy as np
            image = image.squeeze().permute(1, 2, 0).cpu().numpy()
            image = (image * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(image)

        return GenerationResult(image=image.convert("RGB"), prompt=prompt)

    def edit(self, image: Image.Image, original_prompt: str, edit_instruction: str, **kwargs) -> GenerationResult:
        """BAGEL: no native editing, regenerate with combined prompt."""
        combined = f"{original_prompt}. {edit_instruction}"
        return self.generate(combined, **kwargs)
