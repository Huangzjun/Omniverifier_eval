#!/usr/bin/env python3
"""Run OmniVerifier-TTS evaluation on GenEval++.

This script implements the full pipeline for GenEval++:
1. Load step-0 images (pre-generated)
2. Run sequential OmniVerifier-TTS refinement
3. Evaluate final images using VQAScore
4. Compute per-category and overall scores

Usage:
    python scripts/run_geneval_plus.py \
        --config configs/geneval_plus.yaml \
        --step0_dir results/step0/geneval_plus/images \
        --max_rounds 3 \
        --output_dir results/tts/geneval_plus
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
from utils.metrics import aggregate_scores, format_results_table


def parse_args():
    parser = argparse.ArgumentParser(description="OmniVerifier-TTS on GenEval++")
    parser.add_argument("--config", type=str, default="configs/geneval_plus.yaml")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--step0_dir", type=str, required=True, help="Directory with step-0 images")
    parser.add_argument("--max_rounds", type=int, default=3, help="Max TTS refinement rounds")
    parser.add_argument("--output_dir", type=str, default="results/tts/geneval_plus")
    parser.add_argument("--skip_tts", action="store_true", help="Skip TTS, evaluate step-0 directly")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing results")
    parser.add_argument("--generator", type=str, default="qwen_image")
    parser.add_argument("--batch_size", type=int, default=8, help="Eval batch size for VQAScore")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("geneval_plus", log_file=str(output_dir / "eval.log"))

    # Load configs
    base_cfg = load_yaml(args.base_config) if Path(args.base_config).exists() else {}
    bench_cfg = load_yaml(args.config)

    # Build dataset
    data_cfg = bench_cfg.get("data", {})
    dataset = build_dataset("geneval_plus", **data_cfg)
    dataset.load()
    logger.info(f"Loaded {len(dataset)} GenEval++ prompts")

    # Load step-0 images
    step0_path = Path(args.step0_dir)
    step0_images = {}
    for sample in dataset:
        img_path = step0_path / f"{sample.id}.png"
        if img_path.exists():
            step0_images[sample.id] = load_image(img_path)
    logger.info(f"Loaded {len(step0_images)} step-0 images")

    if not args.eval_only and not args.skip_tts:
        # === Phase 1: OmniVerifier-TTS Refinement ===
        logger.info("=" * 60)
        logger.info("Phase 1: OmniVerifier-TTS Sequential Refinement")
        logger.info("=" * 60)

        model_cfg = base_cfg.get("model", {}).get("omniverifier", {})
        verifier = OmniVerifier(**model_cfg)
        verifier.load()

        generator = build_generator(args.generator)

        tts = SequentialTTS(
            verifier=verifier,
            generator=generator,
            max_rounds=args.max_rounds,
            early_stop=True,
            save_intermediates=True,
            output_dir=str(output_dir),
        )

        tts_results = []
        for sample in tqdm(dataset, desc="Running OmniVerifier-TTS"):
            initial_image = step0_images.get(sample.id)
            if initial_image is None:
                continue

            result = tts.run(
                sample_id=sample.id,
                prompt=sample.prompt,
                initial_image=initial_image,
            )
            tts_results.append({
                "id": sample.id,
                "prompt": sample.prompt,
                "category": sample.dimension,
                "final_step": result.final_step,
                "is_aligned": result.is_aligned,
            })

        save_json(tts_results, output_dir / "tts_results.json")

    # === Phase 2: VQAScore Evaluation ===
    logger.info("=" * 60)
    logger.info("Phase 2: GenEval++ VQAScore Evaluation")
    logger.info("=" * 60)

    eval_cfg = bench_cfg.get("evaluation", {})
    evaluator = build_evaluator(
        "geneval_plus",
        model_name=eval_cfg.get("vqascore_model", "clip-flant5-xxl"),
    )
    evaluator.load()

    # Determine images to evaluate
    if args.skip_tts or args.eval_only:
        images_to_eval = step0_images
        image_source = "step0"
    else:
        final_dir = output_dir / "final"
        images_to_eval = {}
        for sample in dataset:
            final_path = final_dir / f"{sample.id}.png"
            if final_path.exists():
                images_to_eval[sample.id] = load_image(final_path)
            elif sample.id in step0_images:
                images_to_eval[sample.id] = step0_images[sample.id]
        image_source = "tts_final"

    logger.info(f"Evaluating {len(images_to_eval)} images (source: {image_source})")

    # Batch evaluation for efficiency
    eval_results = []
    batch_images, batch_prompts, batch_ids, batch_metas = [], [], [], []

    for sample in dataset:
        if sample.id not in images_to_eval:
            continue

        batch_images.append(images_to_eval[sample.id])
        batch_prompts.append(sample.prompt)
        batch_ids.append(sample.id)
        batch_metas.append(sample.metadata)

        if len(batch_images) >= args.batch_size:
            results = evaluator.evaluate_batch(batch_images, batch_prompts, batch_ids, batch_metas)
            eval_results.extend(results)
            batch_images, batch_prompts, batch_ids, batch_metas = [], [], [], []

    # Process remaining batch
    if batch_images:
        results = evaluator.evaluate_batch(batch_images, batch_prompts, batch_ids, batch_metas)
        eval_results.extend(results)

    # Compute aggregate scores
    results_dicts = [
        {"sample_id": r.sample_id, "score": r.score, "category": r.dimension, "details": r.details}
        for r in eval_results
    ]

    scores = aggregate_scores(results_dicts)

    logger.info("\n" + "=" * 60)
    logger.info("GenEval++ Results")
    logger.info("=" * 60)
    logger.info(f"\nImage source: {image_source}")
    logger.info(format_results_table(scores))

    save_json({
        "benchmark": "geneval_plus",
        "image_source": image_source,
        "max_rounds": args.max_rounds,
        "aggregate_scores": scores,
        "per_sample_results": results_dicts,
    }, output_dir / f"eval_results_{image_source}.json")

    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Overall Score: {scores['overall']:.2f}%")


if __name__ == "__main__":
    main()
