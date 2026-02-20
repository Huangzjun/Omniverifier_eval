#!/usr/bin/env python3
"""Analyze and display results in Table 3 format.

Reads evaluation results from multiple directories and compiles
them into the final Table 3 format.

Usage:
    python scripts/analyze_results.py \
        --results_dirs results/tts/t2i_reasonbench results/tts/geneval_plus

    # Or from a full eval run:
    python scripts/analyze_results.py \
        --full_eval_dir results/full_eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze OmniVerifier-TTS results")
    parser.add_argument("--results_dirs", nargs="*", help="Individual result directories")
    parser.add_argument("--full_eval_dir", type=str, help="Full eval output directory")
    parser.add_argument("--output", type=str, default=None, help="Output file for summary")
    return parser.parse_args()


def load_results(path: Path) -> dict:
    """Load results JSON from a directory."""
    candidates = [
        path / "table3_results.json",
        path / "all_results.json",
        path / "eval_results_tts_final.json",
        path / "eval_results_step0.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c) as f:
                return json.load(f)
    return {}


def print_table3(results: dict):
    """Print results in Table 3 format."""
    print("\n" + "=" * 80)
    print("TABLE 3: Evaluation of OmniVerifier-TTS on reasoning and compositional generation benchmarks")
    print("=" * 80)

    # Check if we have full eval format or individual benchmark format
    if "t2i_reasonbench" in results or "geneval_plus" in results:
        # Full eval format
        print(f"\n{'Method':<45} {'T2I-ReasonBench':>16} {'GenEval++':>12}")
        print("-" * 75)

        conditions = set()
        for bench_results in results.values():
            if isinstance(bench_results, dict):
                conditions.update(bench_results.keys())

        labels = {
            "step0": "Base Model (Step 0)",
            "sequential_tts": "+ OmniVerifier-TTS (Sequential)",
            "parallel_tts_n4": "+ Best-of-4 (Parallel)",
        }

        for cond in ["step0", "sequential_tts", "parallel_tts_n4"]:
            if cond not in conditions:
                continue

            label = labels.get(cond, cond)
            r = results.get("t2i_reasonbench", {}).get(cond, {})
            g = results.get("geneval_plus", {}).get(cond, {})

            r_score = r.get("overall", "-") if isinstance(r, dict) else "-"
            g_score = g.get("overall", "-") if isinstance(g, dict) else "-"

            r_str = f"{r_score:.1f}" if isinstance(r_score, (int, float)) else str(r_score)
            g_str = f"{g_score:.1f}" if isinstance(g_score, (int, float)) else str(g_score)

            print(f"{label:<45} {r_str:>16} {g_str:>12}")

        # Print per-dimension breakdown for T2I-ReasonBench
        t2i_results = results.get("t2i_reasonbench", {})
        if t2i_results:
            print("\n\nT2I-ReasonBench Per-Dimension Breakdown:")
            print("-" * 75)
            print(f"{'Dimension':<30}", end="")
            for cond in ["step0", "sequential_tts"]:
                if cond in t2i_results:
                    label = "Step 0" if cond == "step0" else "Sequential TTS"
                    print(f" {label:>20}", end="")
            print()
            print("-" * 75)

            dims = ["idiom_interpretation", "textual_image_design", "entity_reasoning", "scientific_reasoning"]
            for dim in dims:
                print(f"{dim:<30}", end="")
                for cond in ["step0", "sequential_tts"]:
                    if cond in t2i_results:
                        per_cat = t2i_results[cond].get("per_category", t2i_results[cond])
                        score = per_cat.get(dim, "-")
                        s = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
                        print(f" {s:>20}", end="")
                print()

    else:
        # Individual benchmark format
        print(json.dumps(results, indent=2))

    print("\n" + "=" * 80)

    # Paper reference values
    print("\nReference values from paper:")
    print("-" * 75)
    print(f"{'Qwen-Image (Step 0)':<45} {'55.5':>16} {'73.1':>12}")
    print(f"{'+ OmniVerifier-TTS (Sequential)':<45} {'59.2':>16} {'77.4':>12}")
    print(f"{'GPT-Image-1 (Step 0)':<45} {'76.8':>16} {'81.4':>12}")
    print(f"{'+ OmniVerifier-TTS (Sequential)':<45} {'79.3':>16} {'85.7':>12}")


def main():
    args = parse_args()

    all_results = {}

    if args.full_eval_dir:
        full_path = Path(args.full_eval_dir)
        all_results = load_results(full_path)
    elif args.results_dirs:
        for result_dir in args.results_dirs:
            path = Path(result_dir)
            data = load_results(path)
            if data:
                # Infer benchmark name from directory
                bench = path.name
                if "benchmark" in data:
                    bench = data["benchmark"]
                all_results[bench] = data
    else:
        print("Error: Provide either --full_eval_dir or --results_dirs")
        sys.exit(1)

    print_table3(all_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved to {args.output}")


if __name__ == "__main__":
    main()
