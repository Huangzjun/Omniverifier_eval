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
        print(f"[JanusPro] Loading model from {self.model_path} ...")

        from janus.models import MultiModalityCausalLM, VLChatProcessor

        self._processor = VLChatProcessor.from_pretrained(self.model_path)
        self._tokenizer = self._processor.tokenizer
        self._model = MultiModalityCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
        ).to(self.device).eval()

        print("[JanusPro] Model loaded successfully")

    @torch.inference_mode()
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate image using Janus-Pro official generation pipeline."""
        import numpy as np

        if self._model is None:
            self.load()

        temperature = kwargs.get("temperature", 1.0)
        cfg_weight = kwargs.get("cfg_weight", 5.0)
        image_token_num = kwargs.get("image_token_num", 576)
        img_size = kwargs.get("img_size", 384)
        patch_size = kwargs.get("patch_size", 16)

        conversation = [
            {"role": "User", "content": prompt},
            {"role": "Assistant", "content": ""},
        ]
        sft_format = self._processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self._processor.sft_format,
            system_prompt="",
        )
        prompt_text = sft_format + self._processor.image_start_tag

        input_ids = self._processor.tokenizer.encode(prompt_text)
        input_ids = torch.LongTensor(input_ids)

        # CFG: paired conditional / unconditional tokens
        tokens = torch.zeros((2, len(input_ids)), dtype=torch.int, device=self.device)
        tokens[0, :] = input_ids
        tokens[1, :] = input_ids
        tokens[1, 1:-1] = self._processor.pad_id

        inputs_embeds = self._model.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((1, image_token_num), dtype=torch.int, device=self.device)

        past_key_values = None
        for i in range(image_token_num):
            outputs = self._model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state

            logits = self._model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0:1, :]
            logit_uncond = logits[1:2, :]

            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token_paired = torch.cat([next_token, next_token], dim=0).view(-1)
            img_embeds = self._model.prepare_gen_img_embeds(next_token_paired)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = self._model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[1, 8, img_size // patch_size, img_size // patch_size],
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

        image = Image.fromarray(dec[0])
        return GenerationResult(image=image.convert("RGB"), prompt=prompt)

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
        model_path: str = "models/bagel-7b",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        use_thinking: bool = True,
        bagel_repo: str = "bagel_repo",
    ):
        super().__init__(name="bagel")
        self.model_path = model_path
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.use_thinking = use_thinking
        self.bagel_repo = bagel_repo
        self._inferencer = None

    def load(self) -> None:
        """Load BAGEL model using official pipeline."""
        import sys, os
        from pathlib import Path

        repo_root = str(Path(__file__).parent.parent / self.bagel_repo)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # The project has its own `data` package at the repo root which
        # shadows `bagel_repo/data`.  Temporarily evict all cached `data.*`
        # entries so the imports below resolve against bagel_repo.
        _saved_data_modules = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "data" or k.startswith("data.")
        }

        print(f"[BAGEL] Loading model from {self.model_path} ...")

        try:
            from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
            from data.data_utils import add_special_tokens
            from data.transforms import ImageTransform
            from inferencer import InterleaveInferencer
            from modeling.autoencoder import load_ae
            from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM
            from modeling.bagel import SiglipVisionConfig, SiglipVisionModel
            from modeling.qwen2 import Qwen2Tokenizer
        finally:
            # Restore the project-level `data` package so the rest of the
            # codebase keeps working.
            bagel_data_modules = {
                k: sys.modules.pop(k)
                for k in list(sys.modules)
                if k == "data" or k.startswith("data.")
            }
            sys.modules.update(_saved_data_modules)

        llm_config = Qwen2Config.from_json_file(os.path.join(self.model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(self.model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        vae_model, vae_config = load_ae(local_path=os.path.join(self.model_path, "ae.safetensors"))

        config = BagelConfig(
            visual_gen=True, visual_und=True,
            llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
            vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh',
            latent_patch_size=2, max_latent_size=64,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        device_map = infer_auto_device_map(
            model,
            max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        same_device_modules = [
            'language_model.model.embed_tokens', 'time_embedder',
            'latent_pos_embed', 'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed',
        ]
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device

        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(self.model_path, "ema.safetensors"),
            device_map=device_map, offload_buffers=True,
            dtype=torch.bfloat16, force_hooks=True,
        ).eval()

        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        self._inferencer = InterleaveInferencer(
            model=model, vae_model=vae_model, tokenizer=tokenizer,
            vae_transform=vae_transform, vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )
        print(f"[BAGEL] Model loaded (thinking={'on' if self.use_thinking else 'off'})")

    @torch.inference_mode()
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate image using BAGEL official pipeline."""
        if self._inferencer is None:
            self.load()

        result = self._inferencer(
            text=prompt,
            think=self.use_thinking,
            cfg_text_scale=kwargs.get("cfg_text_scale", 4.0),
            cfg_img_scale=kwargs.get("cfg_img_scale", 1.0),
            cfg_interval=kwargs.get("cfg_interval", [0.4, 1.0]),
            timestep_shift=kwargs.get("timestep_shift", 3.0),
            num_timesteps=kwargs.get("num_timesteps", 50),
            cfg_renorm_min=kwargs.get("cfg_renorm_min", 0.0),
            cfg_renorm_type=kwargs.get("cfg_renorm_type", "global"),
            image_shapes=kwargs.get("image_shapes", (1024, 1024)),
        )
        image = result["image"]
        if not isinstance(image, Image.Image):
            import numpy as np
            image = Image.fromarray(np.uint8(image))
        return GenerationResult(image=image.convert("RGB"), prompt=prompt)

    def edit(self, image: Image.Image, original_prompt: str, edit_instruction: str, **kwargs) -> GenerationResult:
        """BAGEL: no native editing, regenerate with combined prompt."""
        combined = f"{original_prompt}. {edit_instruction}"
        return self.generate(combined, **kwargs)
