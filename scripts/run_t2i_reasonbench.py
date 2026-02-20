#!/usr/bin/env python3
"""Run OmniVerifier-TTS evaluation on T2I-ReasonBench.

This script implements the full pipeline:
1. Load step-0 images (pre-generated)
2. Run sequential OmniVerifier-TTS refinement
3. Evaluate final images using T2I-ReasonBench protocol
4. Compute per-dimension and overall scores

Usage:
    python scripts/run_t2i_reasonbench.py \
        --config configs/t2i_reasonbench.yaml \
        --step0_dir results/step0/t2i_reasonbench/images \
        --max_rounds 3 \
        --output_dir results/tts/t2i_reasonbench
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from data import build_dataset
from evaluators import build_evaluator
from generators import build_generator
from pipeline import OmniVerifier, SequentialTTS
from utils.io_utils import load_yaml, load_image, save_json, ensure_dir
from utils.logger import setup_logger
from utils.metrics import compute_dimension_scores, format_results_table


def parse_args():
    parser = argparse.ArgumentParser(description="OmniVerifier-TTS on T2I-ReasonBench")
    parser.add_argument("--config", type=str, default="configs/t2i_reasonbench.yaml")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--step0_dir", type=str, required=True, help="Directory with step-0 images")
    parser.add_argument("--max_rounds", type=int, default=3, help="Max TTS refinement rounds")
    parser.add_argument("--output_dir", type=str, default="results/tts/t2i_reasonbench")
    parser.add_argument("--skip_tts", action="store_true", help="Skip TTS, evaluate step-0 directly")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation on existing TTS results")
    parser.add_argument("--generator", type=str, default="qwen_image")
    return parser.parse_args()


def load_step0_images(step0_dir: str, dataset) -> dict[str, any]:
    """Load pre-generated step-0 images."""
    from PIL import Image

    step0_path = Path(step0_dir)
    images = {}

    for sample in dataset:
        # Try multiple naming conventions
        candidates = [
            step0_path / f"{sample.id}.png",
            step0_path / f"{sample.metadata.get('index', 0) + 1:04d}.png",
            step0_path / sample.dimension / f"{sample.metadata.get('index', 0) + 1:04d}.png",
        ]

        for path in candidates:
            if path.exists():
                images[sample.id] = Image.open(path).convert("RGB")
                break

    return images


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("t2i_reasonbench", log_file=str(output_dir / "eval.log"))

    # Load configs
    base_cfg = load_yaml(args.base_config) if Path(args.base_config).exists() else {}
    bench_cfg = load_yaml(args.config)

    # Build dataset
    data_cfg = bench_cfg.get("data", {})
    dataset = build_dataset("t2i_reasonbench", **data_cfg)
    dataset.load()

    dimensions = dataset.get_dimensions()
    logger.info(f"Loaded {len(dataset)} prompts across dimensions: {dimensions}")

    # Load step-0 images
    step0_images = load_step0_images(args.step0_dir, dataset)
    logger.info(f"Loaded {len(step0_images)} step-0 images")

    if not args.eval_only:
        # === Phase 1: OmniVerifier-TTS Refinement ===
        if not args.skip_tts:
            logger.info("=" * 60)
            logger.info("Phase 1: OmniVerifier-TTS Sequential Refinement")
            logger.info("=" * 60)

            # Load OmniVerifier
            model_cfg = base_cfg.get("model", {}).get("omniverifier", {})
            verifier = OmniVerifier(**model_cfg)
            verifier.load()

            # Load generator
            generator = build_generator(args.generator)

            # Build TTS pipeline
            tts = SequentialTTS(
                verifier=verifier,
                generator=generator,
                max_rounds=args.max_rounds,
                early_stop=True,
                save_intermediates=True,
                output_dir=str(output_dir),
            )

            # Run TTS on all samples
            tts_results = []
            for i, sample in enumerate(tqdm(dataset, desc="Running OmniVerifier-TTS")):
                initial_image = step0_images.get(sample.id)
                if initial_image is None:
                    logger.warning(f"No step-0 image for {sample.id}, skipping")
                    continue

                result = tts.run(
                    sample_id=sample.id,
                    prompt=sample.prompt,
                    initial_image=initial_image,
                )
                tts_results.append({
                    "id": sample.id,
                    "prompt": sample.prompt,
                    "dimension": sample.dimension,
                    "final_step": result.final_step,
                    "is_aligned": result.is_aligned,
                    "total_time": result.total_time,
                })

            save_json(tts_results, output_dir / "tts_results.json")
            logger.info(f"TTS refinement complete. {len(tts_results)} samples processed.")

    # === Phase 2: Evaluation ===
    logger.info("=" * 60)
    logger.info("Phase 2: T2I-ReasonBench Evaluation")
    logger.info("=" * 60)

    # Load evaluator
    eval_cfg = bench_cfg.get("evaluation", {})
    evaluator = build_evaluator(
        "t2i_reasonbench",
        model_path=eval_cfg.get("judge_model", "Qwen/Qwen2.5-VL-7B-Instruct"),
    )
    evaluator.load()

    # Determine which images to evaluate
    if args.skip_tts or args.eval_only:
        # Evaluate step-0 images directly
        image_source = "step0"
        images_to_eval = step0_images
    else:
        # Evaluate TTS-refined final images
        image_source = "tts_final"
        final_dir = output_dir / "final"
        images_to_eval = {}
        for sample in dataset:
            final_path = final_dir / f"{sample.id}.png"
            if final_path.exists():
                images_to_eval[sample.id] = load_image(final_path)
            elif sample.id in step0_images:
                # Fallback to step-0 if TTS didn't produce a result
                images_to_eval[sample.id] = step0_images[sample.id]

    logger.info(f"Evaluating {len(images_to_eval)} images (source: {image_source})")

    # Run evaluation
    eval_results = []
    for sample in tqdm(dataset, desc="Evaluating"):
        if sample.id not in images_to_eval:
            continue

        image = images_to_eval[sample.id]
        result = evaluator.evaluate_single(
            image=image,
            prompt=sample.prompt,
            metadata={**sample.metadata, "dimension_full_name": sample.dimension},
        )
        result.sample_id = sample.id
        result.dimension = sample.dimension
        eval_results.append(result)

    # Compute scores
    results_dicts = [
        {"sample_id": r.sample_id, "score": r.score, "dimension": r.dimension, "details": r.details}
        for r in eval_results
    ]

    dimension_scores = compute_dimension_scores(results_dicts, dimensions)

    logger.info("\n" + "=" * 60)
    logger.info("T2I-ReasonBench Results")
    logger.info("=" * 60)
    logger.info(f"\nImage source: {image_source}")
    logger.info(format_results_table(dimension_scores))

    # Save results
    save_json({
        "benchmark": "t2i_reasonbench",
        "image_source": image_source,
        "max_rounds": args.max_rounds,
        "dimension_scores": dimension_scores,
        "per_sample_results": results_dicts,
    }, output_dir / f"eval_results_{image_source}.json")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Overall Score: {dimension_scores['overall']:.2f}%")


if __name__ == "__main__":
    main()
