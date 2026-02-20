from .base_evaluator import BaseEvaluator, EvalResult
from .t2i_reasonbench_eval import T2IReasonBenchEvaluator
from .geneval_plus_eval import GenEvalPlusEvaluator

EVALUATOR_REGISTRY = {
    "t2i_reasonbench": T2IReasonBenchEvaluator,
    "geneval_plus": GenEvalPlusEvaluator,
}


def build_evaluator(name: str, **kwargs) -> BaseEvaluator:
    """Factory function to build evaluator by name."""
    if name not in EVALUATOR_REGISTRY:
        raise ValueError(f"Unknown evaluator: {name}. Available: {list(EVALUATOR_REGISTRY.keys())}")
    return EVALUATOR_REGISTRY[name](**kwargs)
