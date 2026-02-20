from .base_dataset import BaseDataset, DataSample
from .t2i_reasonbench import T2IReasonBenchDataset
from .geneval_plus import GenEvalPlusDataset

DATASET_REGISTRY = {
    "t2i_reasonbench": T2IReasonBenchDataset,
    "geneval_plus": GenEvalPlusDataset,
}


def build_dataset(name: str, **kwargs) -> BaseDataset:
    """Factory function to build dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](**kwargs)
