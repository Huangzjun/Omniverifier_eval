#!/usr/bin/env python3
"""Run baseline (step-0) evaluation across multiple T2I models.

This produces the comparison rows in Table 3 that include
diffusion baselines (SD3, Flux, SDXL) alongside UMMs.

For pure T2I models, only step-0 scores are reported (no TTS refinement).
For UMMs (Qwen-Image, GPT-Image-1), both step-0 and TTS scores are reported.

Usage:
    # Evaluate specific diffusion models on T2I-ReasonBench
    python scripts/run_baselines.py \
        --benchmark t2i_reasonbench \
        --models flux-dev sd3.5-large sdxl \
        --output_dir results/baselines

    # Evaluate all available models
    python scripts/run_baselines.py \
        --benchmark t2i_reasonbench \
        --models all \
        --output_dir results/baselines
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from data import build_dataset
from evaluators import build_evaluator
from generators import build_generator, DIFFUSION_PRESETS, GENERATOR_REGISTRY
from utils.io_utils import load_yaml, save_json, save_image, ensure_dir
from utils.logger import setup_logger
from utils.metrics import aggregate_scores, compute_dimension_scores


ALL_MODELS = list(GENERATOR_REGISTRY.keys()) + DIFFUSION_PRESETS


def parse_args():
    parser = argparse.ArgumentParser(description="Run baselines across multiple T2I models")
    parser.add_argument("--benchmark", type=str, required=True, choices=["t2i_reasonbench", "geneval_plus"])
    parser.add_argument("--models", nargs="+", default=["flux-dev", "sd3.5-large"],
                        help=f"Model names to evaluate. Use 'all' for all. Available: {ALL_MODELS}")
    parser.add_argument("--output_dir", type=str, default="results/baselines")
    parser.add_argument("--num_samples", type=int, default=-1, help="Limit number of samples (-1 for all)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("baselines", log_file=str(output_dir / "baselines.log"))

    # Resolve model list
    if "all" in args.models:
        model_names = ALL_MODELS
    else:
        model_names = args.models

    # Load benchmark dataset
    bench_cfg_path = f"configs/{args.benchmark}.yaml"
    bench_cfg = load_yaml(bench_cfg_path) if Path(bench_cfg_path).exists() else {}
    data_cfg = bench_cfg.get("data", {})

    dataset = build_dataset(args.benchmark, **data_cfg)
    dataset.load()

    samples = list(dataset)
    if args.num_samples > 0:
        samples = samples[:args.num_samples]
    logger.info(f"Evaluating {len(samples)} samples on {args.benchmark}")

    # Load evaluator
    evaluator = build_evaluator(args.benchmark)
    evaluator.load()

    # Run each model
    all_model_results = {}

    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*60}")

        model_dir = ensure_dir(output_dir / model_name / args.benchmark)
        images_dir = ensure_dir(model_dir / "images")

        # Build generator
        try:
            generator = build_generator(model_name)
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            continue

        # Generate images
        generated_images = {}
        for sample in tqdm(samples, desc=f"Generating ({model_name})"):
            img_path = images_dir / f"{sample.id}.png"

            if img_path.exists():
                from PIL import Image
                generated_images[sample.id] = Image.open(img_path).convert("RGB")
                continue

            try:
                result = generator.generate(sample.prompt)
                save_image(result.image, img_path)
                generated_images[sample.id] = result.image
            except Exception as e:
                logger.error(f"  Generation failed for {sample.id}: {e}")

        logger.info(f"  Generated {len(generated_images)} images")

        # Evaluate
        eval_results = []
        for sample in tqdm(samples, desc=f"Evaluating ({model_name})"):
            if sample.id not in generated_images:
                continue

            result = evaluator.evaluate_single(
                generated_images[sample.id],
                sample.prompt,
                sample.metadata,
            )
            result.sample_id = sample.id
            result.dimension = sample.dimension
            eval_results.append(result)

        # Compute scores
        results_dicts = [
            {"sample_id": r.sample_id, "score": r.score, "dimension": r.dimension, "category": r.dimension}
            for r in eval_results
        ]

        if args.benchmark == "t2i_reasonbench":
            scores = compute_dimension_scores(results_dicts, dataset.get_dimensions())
        else:
            scores = aggregate_scores(results_dicts)

        all_model_results[model_name] = scores
        save_json({"scores": scores, "per_sample": results_dicts}, model_dir / "results.json")

        overall = scores.get("overall", 0)
        logger.info(f"  {model_name} overall: {overall:.2f}%")

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info(f"BASELINE COMPARISON: {args.benchmark}")
    logger.info("=" * 80)

    # Sort by overall score
    sorted_models = sorted(
        all_model_results.items(),
        key=lambda x: x[1].get("overall", 0),
        reverse=True,
    )

    logger.info(f"\n{'Rank':<6} {'Model':<25} {'Overall':>10}")
    logger.info("-" * 45)
    for rank, (model, scores) in enumerate(sorted_models, 1):
        overall = scores.get("overall", 0)
        logger.info(f"{rank:<6} {model:<25} {overall:>10.1f}%")

    # Dimension breakdown for T2I-ReasonBench
    if args.benchmark == "t2i_reasonbench":
        dims = ["idiom_interpretation", "textual_image_design", "entity_reasoning", "scientific_reasoning"]
        logger.info(f"\n{'Model':<20}", end="")
        for d in dims:
            short = d[:12]
            logger.info(f" {short:>13}", end="")
        logger.info(f" {'Overall':>10}")
        logger.info("-" * (20 + 13 * len(dims) + 10))

        for model, scores in sorted_models:
            logger.info(f"{model:<20}", end="")
            per_cat = scores.get("per_category", scores)
            for d in dims:
                val = per_cat.get(d, scores.get(d, 0))
                logger.info(f" {val:>13.1f}", end="")
            logger.info(f" {scores.get('overall', 0):>10.1f}")

    save_json(all_model_results, output_dir / f"baselines_{args.benchmark}.json")
    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
