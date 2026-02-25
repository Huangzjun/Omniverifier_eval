#!/usr/bin/env python3
"""Reproduce Table 3: OmniVerifier-TTS evaluation.

This script runs all 10 experimental conditions for Table 3:

  ┌──────────────────────────────────────────────────────────────────┐
  │  #  │ Condition                         │ Generator  │ Verifier │
  ├──────────────────────────────────────────────────────────────────┤
  │  1  │ Janus-Pro (step 0)                │ janus_pro  │   —      │
  │  2  │ BAGEL (step 0)                    │ bagel      │   —      │
  │  3  │ SD-3-Medium (step 0)              │ sd3_medium │   —      │
  │  4  │ FLUX.1-dev (step 0)               │ flux_dev   │   —      │
  │  5  │ Qwen-Image (step 0)               │ qwen_image │   —      │
  │  6  │ GPT-Image-1 (step 0)              │ gpt_image  │   —      │
  │  7  │ QwenVL-TTS (Qwen-Image)           │ qwen_image │ qwenvl   │
  │  8  │ OmniVerifier-TTS (Qwen-Image)     │ qwen_image │ omniver  │
  │  9  │ QwenVL-TTS (GPT-Image-1)          │ gpt_image  │ qwenvl   │
  │ 10  │ OmniVerifier-TTS (GPT-Image-1)    │ gpt_image  │ omniver  │
  └──────────────────────────────────────────────────────────────────┘

Conditions 1-6: Generate images → evaluate directly (no TTS loop).
Conditions 7-10: Reuse step-0 images from conditions 5/6 → run
                 sequential TTS verify→edit loop → evaluate final images.

Benchmarks: T2I-ReasonBench (800 prompts) and GenEval++ (VQAScore).

Usage:
    # Run everything
    python scripts/run_table3.py --output_dir results/table3

    # Run only specific conditions
    python scripts/run_table3.py --conditions 1 2 5 8 10

    # Run only specific benchmark
    python scripts/run_table3.py --benchmark t2i_reasonbench

    # Limit samples for debugging
    python scripts/run_table3.py --num_samples 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from tqdm import tqdm

from data import build_dataset
from evaluators import build_evaluator
from generators import build_generator, TTS_CAPABLE
from pipeline import build_verifier, SequentialTTS
from utils.io_utils import save_json, save_image, ensure_dir, load_yaml
from utils.logger import setup_logger
from utils.metrics import aggregate_scores, compute_dimension_scores


# ═══════════════════════════════════════════════════════════════════
#  Condition definitions
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Condition:
    """One experimental condition in Table 3."""
    id: int
    name: str               # Display name in the table
    generator: str           # Key in GENERATOR_REGISTRY
    verifier: str | None     # Key in VERIFIER_REGISTRY, None = step-0 only
    tts_rounds: int = 0      # 0 = step-0 only, >0 = TTS loop
    step0_from: int | None = None  # Reuse step-0 images from this condition id


# fmt: off
CONDITIONS = [
    # ── Step-0 only baselines ────────────────────────────────────
    Condition(id=1,  name="Janus-Pro",                     generator="janus_pro",  verifier=None),
    Condition(id=2,  name="BAGEL",                         generator="bagel",      verifier=None),
    Condition(id=3,  name="SD-3-Medium",                   generator="sd3_medium", verifier=None),
    Condition(id=4,  name="FLUX.1-dev",                    generator="flux_dev",   verifier=None),
    Condition(id=5,  name="Qwen-Image",                    generator="qwen_image", verifier=None),
    Condition(id=6,  name="GPT-Image-1",                   generator="gpt_image",  verifier=None),
    # ── TTS loop conditions ──────────────────────────────────────
    Condition(id=7,  name="QwenVL-TTS (Qwen-Image)",       generator="qwen_image", verifier="qwenvl",       tts_rounds=3, step0_from=5),
    Condition(id=8,  name="OmniVerifier-TTS (Qwen-Image)", generator="qwen_image", verifier="omniverifier", tts_rounds=3, step0_from=5),
    Condition(id=9,  name="QwenVL-TTS (GPT-Image-1)",      generator="gpt_image",  verifier="qwenvl",       tts_rounds=3, step0_from=6),
    Condition(id=10, name="OmniVerifier-TTS (GPT-Image-1)",generator="gpt_image",  verifier="omniverifier", tts_rounds=3, step0_from=6),
]
# fmt: on

CONDITIONS_BY_ID = {c.id: c for c in CONDITIONS}


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

class LazyImageDir(dict):
    """Dict-like that lazily loads images from a directory on access.

    Avoids loading all images into memory at once — each image is read
    from disk when accessed and NOT cached, so memory stays low.
    """

    def __init__(self, images_dir: Path):
        super().__init__()
        self._dir = images_dir
        self._ids = {p.stem for p in images_dir.glob("*.png")} if images_dir.exists() else set()

    def __contains__(self, key):
        return key in self._ids

    def __getitem__(self, key):
        if key not in self._ids:
            raise KeyError(key)
        return Image.open(self._dir / f"{key}.png").convert("RGB")

    def __len__(self):
        return len(self._ids)

    def __iter__(self):
        return iter(self._ids)

    def get(self, key, default=None):
        if key in self._ids:
            return self[key]
        return default

    def keys(self):
        return self._ids


def load_step0_images(images_dir: Path) -> LazyImageDir:
    """Return a lazy-loading dict backed by image files on disk."""
    return LazyImageDir(images_dir)


def run_step0_generation(
    cond: Condition,
    samples: list,
    output_dir: Path,
    logger,
    batch_size: int = 4,
    num_workers: int = 1,
) -> dict[str, Image.Image]:
    """Generate step-0 images for a condition.

    For diffusion models, generates in batches of `batch_size` for GPU parallelism.
    For API-based generators, uses `num_workers` threads for concurrent requests.
    Already-generated images are skipped automatically.
    """
    images_dir = ensure_dir(output_dir / f"cond{cond.id}_{cond.generator}" / "images")

    # Check for existing images (skip already generated)
    existing_ids = {p.stem for p in images_dir.glob("*.png")}
    pending = [s for s in samples if s.id not in existing_ids]

    if not pending:
        logger.info(f"  [{cond.name}] All {len(samples)} images already exist, skipping generation")
        return LazyImageDir(images_dir)

    logger.info(f"  [{cond.name}] {len(existing_ids)} cached, {len(pending)} to generate")

    # Build generator
    logger.info(f"  [{cond.name}] Building generator: {cond.generator}")
    generator = build_generator(cond.generator)

    from generators.diffusion_models import DiffusionGenerator
    use_batch = isinstance(generator, DiffusionGenerator) and batch_size > 1

    if use_batch:
        logger.info(f"  [{cond.name}] Batch mode enabled (batch_size={batch_size})")
        for i in tqdm(range(0, len(pending), batch_size), desc=f"Gen {cond.name} (batch={batch_size})"):
            batch_samples = pending[i : i + batch_size]
            prompts = [s.prompt for s in batch_samples]
            try:
                results = generator.generate_batch(prompts)
                for sample, result in zip(batch_samples, results):
                    save_image(result.image, images_dir / f"{sample.id}.png")
            except Exception as e:
                logger.error(f"  [{cond.name}] Batch generation failed, falling back to single: {e}")
                for sample in batch_samples:
                    try:
                        result = generator.generate(sample.prompt)
                        save_image(result.image, images_dir / f"{sample.id}.png")
                    except Exception as e2:
                        logger.error(f"  [{cond.name}] Generation failed for {sample.id}: {e2}")
    elif num_workers > 1:
        # Concurrent API requests via thread pool
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        logger.info(f"  [{cond.name}] Parallel API mode (workers={num_workers})")
        pbar = tqdm(total=len(pending), desc=f"Gen {cond.name} (workers={num_workers})")
        lock = threading.Lock()

        def _generate_one(sample):
            img_path = images_dir / f"{sample.id}.png"
            if img_path.exists():
                return sample.id, True
            try:
                result = generator.generate(sample.prompt)
                save_image(result.image, img_path)
                return sample.id, True
            except Exception as e:
                logger.error(f"  [{cond.name}] Generation failed for {sample.id}: {e}")
                return sample.id, False

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_generate_one, s): s for s in pending}
            for future in as_completed(futures):
                future.result()
                with lock:
                    pbar.update(1)
        pbar.close()
    else:
        for sample in tqdm(pending, desc=f"Gen {cond.name}"):
            try:
                result = generator.generate(sample.prompt)
                save_image(result.image, images_dir / f"{sample.id}.png")
            except Exception as e:
                logger.error(f"  [{cond.name}] Generation failed for {sample.id}: {e}")

    # Free generator GPU memory before evaluation
    del generator
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return lazy-loading dict (images read from disk on demand, not all at once)
    images = LazyImageDir(images_dir)
    logger.info(f"  [{cond.name}] Total: {len(images)} / {len(samples)} images")
    return images


def run_tts_loop(
    cond: Condition,
    samples: list,
    step0_images: dict[str, Image.Image],
    output_dir: Path,
    logger,
) -> dict[str, Image.Image]:
    """Run sequential TTS verify→edit loop."""
    tts_dir = ensure_dir(output_dir / f"cond{cond.id}_{cond.name.replace(' ', '_').replace('(', '').replace(')', '')}")

    # Check for cached final images
    final_dir = tts_dir / "final"
    if final_dir.exists():
        cached = load_step0_images(final_dir)
        if len(cached) >= len(samples):
            logger.info(f"  [{cond.name}] Found {len(cached)} cached TTS results, skipping")
            return cached

    # Build verifier
    logger.info(f"  [{cond.name}] Building verifier: {cond.verifier}")
    verifier = build_verifier(cond.verifier)
    verifier.load()

    # Build generator (for editing)
    logger.info(f"  [{cond.name}] Building generator: {cond.generator}")
    generator = build_generator(cond.generator)

    # Build TTS pipeline
    tts = SequentialTTS(
        verifier=verifier,
        generator=generator,
        max_rounds=cond.tts_rounds,
        early_stop=True,
        save_intermediates=True,
        output_dir=str(tts_dir),
    )

    # Run TTS on each sample
    final_images = {}
    for sample in tqdm(samples, desc=f"TTS {cond.name}"):
        initial_image = step0_images.get(sample.id)
        if initial_image is None:
            logger.warning(f"  [{cond.name}] No step-0 image for {sample.id}, skipping")
            continue

        try:
            result = tts.run(
                sample_id=sample.id,
                prompt=sample.prompt,
                initial_image=initial_image,
            )
            final_images[sample.id] = result.final_image
        except Exception as e:
            logger.error(f"  [{cond.name}] TTS failed for {sample.id}: {e}")
            # Fallback: use step-0 image
            final_images[sample.id] = initial_image

    logger.info(
        f"  [{cond.name}] TTS completed: {len(final_images)} / {len(samples)} samples"
    )
    return final_images


def evaluate_images(
    cond: Condition,
    benchmark: str,
    samples: list,
    images: dict[str, Image.Image],
    output_dir: Path,
    evaluator,
    logger,
) -> dict[str, Any]:
    """Evaluate generated images on a benchmark."""
    results_path = output_dir / f"cond{cond.id}_eval_{benchmark}.json"

    # Check cache
    if results_path.exists():
        logger.info(f"  [{cond.name}] Found cached eval results")
        with open(results_path) as f:
            return json.load(f)

    eval_results = []
    for sample in tqdm(samples, desc=f"Eval {cond.name}"):
        if sample.id not in images:
            continue
        try:
            result = evaluator.evaluate_single(
                images[sample.id],
                sample.prompt,
                sample.metadata,
            )
            eval_results.append({
                "sample_id": sample.id,
                "score": result.score,
                "dimension": getattr(sample, "dimension", ""),
                "category": getattr(sample, "dimension", ""),
            })
        except Exception as e:
            logger.error(f"  [{cond.name}] Eval failed for {sample.id}: {e}")

    # Aggregate
    if benchmark == "t2i_reasonbench":
        dimensions = ["idiom_interpretation", "textual_image_design",
                       "entity_reasoning", "scientific_reasoning"]
        scores = compute_dimension_scores(eval_results, dimensions)
    else:
        scores = aggregate_scores(eval_results)

    output = {"condition": cond.name, "scores": scores, "per_sample": eval_results}
    save_json(output, results_path)
    return output


# ═══════════════════════════════════════════════════════════════════
#  Table formatting
# ═══════════════════════════════════════════════════════════════════

def print_table3(all_results: dict[int, dict], benchmark: str, logger):
    """Print a formatted Table 3."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"  TABLE 3 — {benchmark.upper()}")
    logger.info("=" * 80)

    if benchmark == "t2i_reasonbench":
        dims = ["idiom_interpretation", "textual_image_design",
                "entity_reasoning", "scientific_reasoning"]
        short_dims = ["Idiom", "TextImg", "Entity", "Science"]
        header = f"{'#':<4} {'Condition':<38} "
        header += " ".join(f"{d:>8}" for d in short_dims)
        header += f" {'Overall':>8}"
        logger.info(header)
        logger.info("-" * len(header))

        for cid in sorted(all_results.keys()):
            res = all_results[cid]
            cond = CONDITIONS_BY_ID[cid]
            scores = res.get("scores", {})
            per_cat = scores.get("per_category", scores)

            line = f"{cid:<4} {cond.name:<38} "
            for d in dims:
                val = per_cat.get(d, 0)
                line += f"{val:>8.1f}"
            overall = scores.get("overall", 0)
            line += f" {overall:>8.1f}"
            logger.info(line)

            # Visual separator between step-0 and TTS blocks
            if cid == 6:
                logger.info("-" * len(header))

    else:  # geneval_plus
        header = f"{'#':<4} {'Condition':<38} {'VQAScore':>10}"
        logger.info(header)
        logger.info("-" * len(header))

        for cid in sorted(all_results.keys()):
            res = all_results[cid]
            cond = CONDITIONS_BY_ID[cid]
            scores = res.get("scores", {})
            overall = scores.get("overall", 0)
            logger.info(f"{cid:<4} {cond.name:<38} {overall:>10.1f}")
            if cid == 6:
                logger.info("-" * len(header))

    logger.info("=" * 80)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce Table 3: OmniVerifier-TTS evaluation"
    )
    parser.add_argument(
        "--conditions", type=int, nargs="*", default=None,
        help="Which conditions to run (1-10). Default: all."
    )
    parser.add_argument(
        "--benchmark", type=str, default="both",
        choices=["t2i_reasonbench", "geneval_plus", "both"],
        help="Which benchmark(s) to evaluate on."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/table3",
    )
    parser.add_argument(
        "--num_samples", type=int, default=-1,
        help="Limit samples for debugging (-1 = all)."
    )
    parser.add_argument(
        "--tts_rounds", type=int, default=3,
        help="Max TTS rounds for conditions 7-10."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for diffusion model generation (default: 4)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel workers for API-based generators (default: 1)."
    )
    parser.add_argument(
        "--generate_only", action="store_true",
        help="Only run image generation (Phase 1 & 2), skip evaluation."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("table3", log_file=str(output_dir / "table3.log"))

    # Resolve conditions
    if args.conditions is None:
        run_conds = CONDITIONS
    else:
        run_conds = [CONDITIONS_BY_ID[i] for i in args.conditions if i in CONDITIONS_BY_ID]

    # Override TTS rounds if specified
    for c in run_conds:
        if c.tts_rounds > 0:
            c.tts_rounds = args.tts_rounds

    # Resolve benchmarks
    if args.benchmark == "both":
        benchmarks = ["t2i_reasonbench", "geneval_plus"]
    else:
        benchmarks = [args.benchmark]

    logger.info("=" * 80)
    logger.info("  REPRODUCING TABLE 3: OmniVerifier-TTS")
    logger.info("=" * 80)
    logger.info(f"  Conditions:  {[c.id for c in run_conds]}")
    logger.info(f"  Benchmarks:  {benchmarks}")
    logger.info(f"  TTS rounds:  {args.tts_rounds}")
    logger.info(f"  Output:      {output_dir}")
    logger.info("")

    # ── Per benchmark ────────────────────────────────────────────
    for benchmark in benchmarks:
        logger.info(f"\n{'━' * 80}")
        logger.info(f"  BENCHMARK: {benchmark}")
        logger.info(f"{'━' * 80}")

        # Load dataset
        bench_cfg_path = Path(f"configs/{benchmark}.yaml")
        bench_cfg = load_yaml(str(bench_cfg_path)) if bench_cfg_path.exists() else {}
        data_cfg = bench_cfg.get("data", {})

        dataset = build_dataset(benchmark, **data_cfg)
        dataset.load()
        samples = list(dataset)
        if args.num_samples > 0:
            samples = samples[: args.num_samples]
        logger.info(f"  Loaded {len(samples)} samples")

        bench_dir = ensure_dir(output_dir / benchmark)

        # Load evaluator (skip if generate_only)
        evaluator = None
        if not args.generate_only:
            evaluator = build_evaluator(benchmark)
            evaluator.load()

        # Storage for step-0 images (shared between conditions)
        step0_cache: dict[int, dict[str, Image.Image]] = {}
        all_results: dict[int, dict] = {}

        # ── Phase 1: Generate step-0 images ──────────────────────
        logger.info("\n── Phase 1: Step-0 Generation ──")

        # Determine which step-0 conditions are needed
        needed_step0 = set()
        for c in run_conds:
            if c.step0_from is not None:
                needed_step0.add(c.step0_from)
            if c.verifier is None:
                needed_step0.add(c.id)

        for c in run_conds:
            if c.id not in needed_step0:
                continue
            logger.info(f"\n  Condition {c.id}: {c.name}")
            images = run_step0_generation(c, samples, bench_dir, logger, batch_size=args.batch_size, num_workers=args.num_workers)
            step0_cache[c.id] = images

        # ── Phase 2: Run TTS loops ───────────────────────────────
        logger.info("\n── Phase 2: TTS Verify→Edit Loops ──")

        tts_images: dict[int, dict[str, Image.Image]] = {}
        for c in run_conds:
            if c.verifier is None:
                continue  # skip step-0 only conditions

            logger.info(f"\n  Condition {c.id}: {c.name}")
            logger.info(f"    Verifier:  {c.verifier}")
            logger.info(f"    Generator: {c.generator}")
            logger.info(f"    Rounds:    {c.tts_rounds}")
            logger.info(f"    Step-0 from: condition {c.step0_from}")

            # Get step-0 images to seed the TTS loop
            source_images = step0_cache.get(c.step0_from, {})
            if not source_images:
                logger.error(f"    No step-0 images from condition {c.step0_from}!")
                continue

            final_images = run_tts_loop(c, samples, source_images, bench_dir, logger)
            tts_images[c.id] = final_images

        if args.generate_only:
            total_imgs = sum(len(step0_cache.get(c.id, {})) for c in run_conds if c.id in step0_cache)
            logger.info(f"\n── --generate_only: skipping evaluation. {total_imgs} images generated. ──")
        else:
            # ── Phase 3: Evaluate all conditions ─────────────────────
            logger.info("\n── Phase 3: Evaluation ──")

            for c in run_conds:
                logger.info(f"\n  Evaluating condition {c.id}: {c.name}")

                # Pick the right images
                if c.verifier is None:
                    images = step0_cache.get(c.id, {})
                else:
                    images = tts_images.get(c.id, {})

                if not images:
                    logger.error(f"    No images available for condition {c.id}")
                    continue

                result = evaluate_images(
                    c, benchmark, samples, images, bench_dir, evaluator, logger,
                )
                all_results[c.id] = result

                overall = result.get("scores", {}).get("overall", 0)
                logger.info(f"    → Overall: {overall:.1f}%")

            # ── Print Table 3 ────────────────────────────────────────
            print_table3(all_results, benchmark, logger)

            # Save full results
            save_json(
                {str(k): v for k, v in all_results.items()},
                bench_dir / "table3_results.json",
            )

    logger.info(f"\n✅ All done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
