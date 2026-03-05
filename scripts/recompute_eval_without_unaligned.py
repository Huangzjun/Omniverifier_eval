#!/usr/bin/env python3
"""Recompute evaluation scores by removing samples whose last TTS step is is_aligned=False.

Instead of re-running the expensive judge model, this script:
1. Reads existing evaluation JSON (per_sample results)
2. Reads step metadata to find which samples' last step is is_aligned=False
3. Filters out those samples
4. Recomputes dimension scores and overall score
5. Saves the filtered result as a new JSON

Usage:
    python3 scripts/recompute_eval_without_unaligned.py \
        --eval_json results/table3/t2i_reasonbench/cond8_eval_t2i_reasonbench_undelete.json \
        --steps_dir results/table3/t2i_reasonbench/cond8_OmniVerifier-TTS_Qwen-Image/steps \
        --output results/table3/t2i_reasonbench/cond8_eval_t2i_reasonbench_aligned_only.json

    # Process cond7 as well
    python3 scripts/recompute_eval_without_unaligned.py \
        --eval_json results/table3/t2i_reasonbench/cond7_eval_t2i_reasonbench_undelete.json \
        --steps_dir results/table3/t2i_reasonbench/cond7_QwenVL-TTS_Qwen-Image/steps \
        --output results/table3/t2i_reasonbench/cond7_eval_t2i_reasonbench_aligned_only.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def find_last_step_aligned(steps_dir: Path, sample_id: str) -> bool | None:
    """Check if the last step of a sample is aligned. Returns None if no metadata found."""
    sample_dir = steps_dir / sample_id
    if not sample_dir.exists():
        return None
    metas = sorted(sample_dir.glob("step_*_meta.json"))
    if not metas:
        return None
    with open(metas[-1]) as f:
        meta = json.load(f)
    return meta.get("is_aligned", False)


def compute_dimension_scores(results: list[dict], dimensions: list[str]) -> dict:
    dim_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        dim = r.get("dimension", "unknown")
        if "score" in r:
            dim_scores[dim].append(r["score"])

    output = {}
    for dim in dimensions:
        scores = dim_scores.get(dim, [])
        output[dim] = float(np.mean(scores)) if scores else 0.0

    all_scores = [r["score"] for r in results if "score" in r]
    output["overall"] = float(np.mean(all_scores)) if all_scores else 0.0
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Recompute eval scores after removing unaligned samples"
    )
    parser.add_argument("--eval_json", type=str, required=True,
                        help="Path to existing evaluation JSON file")
    parser.add_argument("--steps_dir", type=str, required=True,
                        help="Path to TTS steps directory (containing per-sample step metadata)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for filtered JSON (default: <input>_aligned_only.json)")
    args = parser.parse_args()

    eval_path = Path(args.eval_json)
    steps_dir = Path(args.steps_dir)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = eval_path.with_name(eval_path.stem + "_aligned_only.json")

    with open(eval_path) as f:
        data = json.load(f)

    per_sample = data.get("per_sample", [])
    original_count = len(per_sample)

    # Classify samples
    kept = []
    removed = []
    no_steps = []

    for entry in per_sample:
        sample_id = entry["sample_id"]
        aligned = find_last_step_aligned(steps_dir, sample_id)

        if aligned is None:
            no_steps.append(sample_id)
            kept.append(entry)
        elif aligned:
            kept.append(entry)
        else:
            removed.append(entry)

    # Recompute scores
    dimensions = ["idiom_interpretation", "textual_image_design",
                   "entity_reasoning", "scientific_reasoning"]
    new_scores = compute_dimension_scores(kept, dimensions)

    # Build output
    output = {
        "condition": data.get("condition", ""),
        "scores": new_scores,
        "filter_stats": {
            "original_samples": original_count,
            "kept_samples": len(kept),
            "removed_samples": len(removed),
            "no_steps_found": len(no_steps),
        },
        "per_sample": kept,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"Input:    {eval_path}")
    print(f"Steps:    {steps_dir}")
    print(f"Output:   {output_path}")
    print()
    print(f"Original samples:  {original_count}")
    print(f"Kept (aligned):    {len(kept)}")
    print(f"Removed (unaligned): {len(removed)}")
    if no_steps:
        print(f"No steps found:    {len(no_steps)}  (kept by default)")
    print()
    print("── Original scores ──")
    orig_scores = data.get("scores", {})
    for k, v in orig_scores.items():
        print(f"  {k:30s} {v:>8.2f}")
    print()
    print("── New scores (aligned only) ──")
    for k, v in new_scores.items():
        print(f"  {k:30s} {v:>8.2f}")


if __name__ == "__main__":
    main()
