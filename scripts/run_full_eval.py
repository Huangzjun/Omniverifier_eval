#!/usr/bin/env python3
"""Full reproduction of OmniVerifier-TTS Table 3.

Runs the complete pipeline for both T2I-ReasonBench and GenEval++ benchmarks:
1. Generate step-0 images
2. Run OmniVerifier-TTS sequential refinement
3. (Optional) Run Best-of-N parallel baseline
4. Evaluate all conditions
5. Produce the final results table

Usage:
    python scripts/run_full_eval.py \
        --generator qwen_image \
        --max_rounds 3 \
        --output_dir results/full_eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import build_dataset
from evaluators import build_evaluator
from generators import build_generator
from pipeline import OmniVerifier, SequentialTTS, ParallelTTS
from utils.io_utils import load_yaml, save_json, ensure_dir
from utils.logger import setup_logger
from utils.metrics import aggregate_scores, compute_dimension_scores, format_results_table


def parse_args():
    parser = argparse.ArgumentParser(description="Full Table 3 reproduction")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--generator", type=str, default="qwen_image")
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--n_parallel", type=int, default=4, help="N for Best-of-N")
    parser.add_argument("--output_dir", type=str, default="results/full_eval")
    parser.add_argument("--run_parallel", action="store_true", help="Also run parallel TTS baseline")
    parser.add_argument("--benchmarks", nargs="+", default=["t2i_reasonbench", "geneval_plus"])
    return parser.parse_args()


def run_benchmark(
    benchmark_name: str,
    base_cfg: dict,
    verifier: OmniVerifier,
    generator,
    max_rounds: int,
    output_dir: Path,
    run_parallel: bool = False,
    n_parallel: int = 4,
):
    """Run full evaluation pipeline for a single benchmark."""
    logger = setup_logger(f"full_eval.{benchmark_name}")

    # Load benchmark config
    config_path = f"configs/{benchmark_name}.yaml"
    bench_cfg = load_yaml(config_path) if Path(config_path).exists() else {}

    # Build dataset
    data_cfg = bench_cfg.get("data", {})
    dataset = build_dataset(benchmark_name, **data_cfg)
    dataset.load()
    logger.info(f"[{benchmark_name}] Loaded {len(dataset)} prompts")

    bench_dir = ensure_dir(output_dir / benchmark_name)
    all_results = {}

    # --- Condition 1: Step 0 (baseline) ---
    logger.info(f"\n{'='*60}\n[{benchmark_name}] Condition 1: Step 0 (baseline)\n{'='*60}")
    step0_dir = ensure_dir(bench_dir / "step0" / "images")

    from tqdm import tqdm
    step0_images = {}
    for sample in tqdm(dataset, desc="Generating step-0"):
        img_path = step0_dir / f"{sample.id}.png"
        if img_path.exists():
            from PIL import Image
            step0_images[sample.id] = Image.open(img_path).convert("RGB")
        else:
            try:
                result = generator.generate(sample.prompt)
                result.image.save(str(img_path))
                step0_images[sample.id] = result.image
            except Exception as e:
                logger.error(f"Generation failed for {sample.id}: {e}")

    # --- Condition 2: Sequential TTS ---
    logger.info(f"\n{'='*60}\n[{benchmark_name}] Condition 2: Sequential TTS\n{'='*60}")
    tts_dir = ensure_dir(bench_dir / "sequential_tts")

    tts = SequentialTTS(
        verifier=verifier,
        generator=generator,
        max_rounds=max_rounds,
        output_dir=str(tts_dir),
    )

    tts_final_images = {}
    for sample in tqdm(dataset, desc="Sequential TTS"):
        if sample.id not in step0_images:
            continue
        try:
            result = tts.run(sample.id, sample.prompt, step0_images[sample.id])
            tts_final_images[sample.id] = result.final_image
        except Exception as e:
            logger.error(f"TTS failed for {sample.id}: {e}")
            tts_final_images[sample.id] = step0_images[sample.id]

    # --- Condition 3: Parallel TTS (optional) ---
    parallel_images = {}
    if run_parallel:
        logger.info(f"\n{'='*60}\n[{benchmark_name}] Condition 3: Parallel TTS (N={n_parallel})\n{'='*60}")

        parallel_tts = ParallelTTS(
            verifier=verifier,
            generator=generator,
            n_candidates=n_parallel,
        )

        for sample in tqdm(dataset, desc="Parallel TTS"):
            try:
                result = parallel_tts.run(sample.id, sample.prompt)
                parallel_images[sample.id] = result.best_image
            except Exception as e:
                logger.error(f"Parallel TTS failed for {sample.id}: {e}")

    # --- Evaluation ---
    logger.info(f"\n{'='*60}\n[{benchmark_name}] Evaluation Phase\n{'='*60}")

    evaluator = build_evaluator(benchmark_name)
    evaluator.load()

    conditions = {"step0": step0_images, "sequential_tts": tts_final_images}
    if run_parallel and parallel_images:
        conditions[f"parallel_tts_n{n_parallel}"] = parallel_images

    for cond_name, images in conditions.items():
        logger.info(f"\nEvaluating condition: {cond_name}")
        eval_results = []

        for sample in tqdm(dataset, desc=f"Eval ({cond_name})"):
            if sample.id not in images:
                continue
            result = evaluator.evaluate_single(images[sample.id], sample.prompt, sample.metadata)
            result.sample_id = sample.id
            result.dimension = sample.dimension
            eval_results.append(result)

        results_dicts = [
            {"sample_id": r.sample_id, "score": r.score, "dimension": r.dimension, "category": r.dimension}
            for r in eval_results
        ]

        if benchmark_name == "t2i_reasonbench":
            scores = compute_dimension_scores(results_dicts, dataset.get_dimensions())
        else:
            scores = aggregate_scores(results_dicts)

        all_results[cond_name] = scores
        logger.info(f"  {cond_name}: {scores.get('overall', 0):.2f}%")

    save_json(all_results, bench_dir / "all_results.json")
    return all_results


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("full_eval", log_file=str(output_dir / "full_eval.log"))

    base_cfg = load_yaml(args.base_config) if Path(args.base_config).exists() else {}

    # Load shared models
    logger.info("Loading OmniVerifier-7B...")
    model_cfg = base_cfg.get("model", {}).get("omniverifier", {})
    verifier = OmniVerifier(**model_cfg)
    verifier.load()

    logger.info(f"Loading generator: {args.generator}...")
    generator = build_generator(args.generator)

    # Run each benchmark
    final_table = {}
    for bench in args.benchmarks:
        logger.info(f"\n{'#'*60}\n# Benchmark: {bench}\n{'#'*60}")
        results = run_benchmark(
            benchmark_name=bench,
            base_cfg=base_cfg,
            verifier=verifier,
            generator=generator,
            max_rounds=args.max_rounds,
            output_dir=output_dir,
            run_parallel=args.run_parallel,
            n_parallel=args.n_parallel,
        )
        final_table[bench] = results

    # Print final Table 3
    logger.info("\n" + "=" * 80)
    logger.info("TABLE 3: OmniVerifier-TTS Evaluation Results")
    logger.info("=" * 80)

    header = f"{'Method':<40} {'T2I-ReasonBench':>16} {'GenEval++':>12}"
    logger.info(header)
    logger.info("-" * 70)

    conditions = ["step0", "sequential_tts"]
    if args.run_parallel:
        conditions.append(f"parallel_tts_n{args.n_parallel}")

    labels = {
        "step0": "Base Model (Step 0)",
        "sequential_tts": "+ OmniVerifier-TTS (Sequential)",
        f"parallel_tts_n{args.n_parallel}": f"+ Best-of-{args.n_parallel} (Parallel)",
    }

    for cond in conditions:
        label = labels.get(cond, cond)
        reason_score = final_table.get("t2i_reasonbench", {}).get(cond, {}).get("overall", "-")
        geneval_score = final_table.get("geneval_plus", {}).get(cond, {}).get("overall", "-")

        r_str = f"{reason_score:.1f}" if isinstance(reason_score, float) else str(reason_score)
        g_str = f"{geneval_score:.1f}" if isinstance(geneval_score, float) else str(geneval_score)
        logger.info(f"{label:<40} {r_str:>16} {g_str:>12}")

    save_json(final_table, output_dir / "table3_results.json")
    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
