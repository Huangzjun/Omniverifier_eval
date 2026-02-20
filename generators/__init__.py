"""Generator registry for Table 3 reproduction.

All 6 generators used:
  Step-0 only (no TTS):
    - janus_pro:   Janus-Pro-7B (autoregressive UMM)
    - bagel:       BAGEL-7B-MoT (autoregressive UMM)
    - sd3_medium:  Stable-Diffusion-3-Medium (diffusion)
    - flux_dev:    FLUX.1-dev (diffusion)

  Step-0 + TTS loop (support editing):
    - qwen_image:  Qwen-Image via DashScope API
    - gpt_image:   GPT-Image-1 via OpenAI API
"""
from .base_generator import BaseGenerator, GenerationResult
from .qwen_image import QwenImageGenerator, OpenAIImageGenerator
from .diffusion_models import DiffusionGenerator
from .umm_models import JanusProGenerator, BAGELGenerator


# ── Full registry ────────────────────────────────────────────────
GENERATOR_REGISTRY = {
    # UMMs with editing capability (for TTS loop)
    "qwen_image": QwenImageGenerator,
    "gpt_image":  OpenAIImageGenerator,
    # Step-0 only baselines
    "janus_pro":  JanusProGenerator,
    "bagel":      BAGELGenerator,
    "sd3_medium": lambda **kw: DiffusionGenerator(model_name="sd3-medium", **kw),
    "flux_dev":   lambda **kw: DiffusionGenerator(model_name="flux-dev", **kw),
}

# Generators that support the verify→edit TTS loop
TTS_CAPABLE = {"qwen_image", "gpt_image"}

# Generators that are step-0 only
STEP0_ONLY = {"janus_pro", "bagel", "sd3_medium", "flux_dev"}


def build_generator(name: str, **kwargs) -> BaseGenerator:
    """Build a generator by name.

    Args:
        name: One of the keys in GENERATOR_REGISTRY.

    Returns:
        Initialized generator instance.
    """
    if name not in GENERATOR_REGISTRY:
        raise ValueError(
            f"Unknown generator: '{name}'.\n"
            f"  TTS-capable (support editing): {sorted(TTS_CAPABLE)}\n"
            f"  Step-0 only:                   {sorted(STEP0_ONLY)}"
        )
    factory = GENERATOR_REGISTRY[name]
    return factory(**kwargs)
