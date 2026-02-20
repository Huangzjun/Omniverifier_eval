from .omniverifier import OmniVerifier, VerificationResult
from .qwenvl_verifier import QwenVLVerifier
from .sequential_tts import SequentialTTS, TTSResult
from .parallel_tts import ParallelTTS

# Verifier registry: verifier_name → class
VERIFIER_REGISTRY = {
    "omniverifier": OmniVerifier,      # RL-finetuned OmniVerifier-7B
    "qwenvl":       QwenVLVerifier,     # Vanilla Qwen2.5-VL-7B-Instruct
}


def build_verifier(name: str, **kwargs):
    """Build a verifier by name."""
    if name not in VERIFIER_REGISTRY:
        raise ValueError(f"Unknown verifier: '{name}'. Available: {list(VERIFIER_REGISTRY.keys())}")
    return VERIFIER_REGISTRY[name](**kwargs)
