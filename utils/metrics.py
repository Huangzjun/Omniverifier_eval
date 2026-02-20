"""Metric aggregation helpers."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def aggregate_scores(results: list[dict], score_key: str = "score") -> dict[str, float]:
    """Compute overall and per-category aggregate scores."""
    all_scores = [r[score_key] for r in results if score_key in r]
    overall = float(np.mean(all_scores)) if all_scores else 0.0

    # Per-category
    cat_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if score_key in r and "category" in r:
            cat_scores[r["category"]].append(r[score_key])

    per_category = {
        cat: float(np.mean(scores)) for cat, scores in cat_scores.items()
    }

    return {"overall": overall, "per_category": per_category, "n_samples": len(all_scores)}


def compute_dimension_scores(results: list[dict], dimensions: list[str], score_key: str = "score") -> dict[str, float]:
    """Compute scores grouped by dimension (for T2I-ReasonBench)."""
    dim_scores: dict[str, list[float]] = defaultdict(list)

    for r in results:
        dim = r.get("dimension", "unknown")
        if score_key in r:
            dim_scores[dim].append(r[score_key])

    output = {}
    for dim in dimensions:
        scores = dim_scores.get(dim, [])
        output[dim] = float(np.mean(scores)) if scores else 0.0

    all_scores = [r[score_key] for r in results if score_key in r]
    output["overall"] = float(np.mean(all_scores)) if all_scores else 0.0
    return output


def format_results_table(results: dict[str, Any]) -> str:
    """Format results as a readable table string."""
    try:
        from prettytable import PrettyTable
    except ImportError:
        # Fallback to simple formatting
        lines = []
        for k, v in results.items():
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for sk, sv in v.items():
                    lines.append(f"  {sk}: {sv:.2f}" if isinstance(sv, float) else f"  {sk}: {sv}")
            else:
                lines.append(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
        return "\n".join(lines)

    table = PrettyTable()
    table.field_names = ["Metric", "Score"]
    for k, v in results.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                table.add_row([f"  {sk}", f"{sv:.2f}" if isinstance(sv, float) else sv])
        else:
            table.add_row([k, f"{v:.2f}" if isinstance(v, float) else v])
    return str(table)
