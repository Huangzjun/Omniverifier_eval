#!/usr/bin/env python3
"""Generate initial (step 0) images for all benchmark prompts.

This is the first step in the OmniVerifier-TTS pipeline.
Generated images are saved to disk and will be used as input
for the sequential refinement process.

Usage:
    python scripts/generate_step0.py \
        --config configs/t2i_reasonbench.yaml \
        --generator qwen_image \
        --output_dir results/step0/t2i_reasonbench
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from data import build_dataset
from generators import build_generator
from utils.io_utils import load_yaml, save_image, save_json, ensure_dir
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Generate step-0 images for benchmarks")
    parser.add_argument("--config", type=str, required=True, help="Benchmark config file")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml", help="Base config")
    parser.add_argument("--generator", type=str, default="qwen_image", help="Generator backend name")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for images")
    parser.add_argument("--start_idx", type=int, default=0, help="Start from this sample index")
    parser.add_argument("--end_idx", type=int, default=-1, help="End at this sample index (-1 for all)")
    parser.add_argument("--api_key", type=str, default=None, help="API key for the generator")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("generate_step0", log_file=f"{args.output_dir}/generation.log")

    # Load configs
    base_cfg = load_yaml(args.base_config) if Path(args.base_config).exists() else {}
    bench_cfg = load_yaml(args.config)

    # Build dataset
    benchmark_name = bench_cfg.get("benchmark", "t2i_reasonbench")
    data_cfg = bench_cfg.get("data", {})
    dataset = build_dataset(benchmark_name, **data_cfg)
    dataset.load()

    logger.info(f"Loaded {len(dataset)} prompts from {benchmark_name}")

    # Build generator
    gen_cfg = base_cfg.get("generation", {})
    generator = build_generator(
        args.generator,
        api_key=args.api_key,
        **{k: v for k, v in gen_cfg.items() if k not in ("backend",)},
    )

    # Setup output
    output_dir = ensure_dir(args.output_dir)
    images_dir = ensure_dir(output_dir / "images")

    # Determine range
    end_idx = args.end_idx if args.end_idx > 0 else len(dataset)
    samples = list(dataset)[args.start_idx:end_idx]

    logger.info(f"Generating images for {len(samples)} samples [{args.start_idx}:{end_idx}]")

    # Generate images
    results_log = []
    for i, sample in enumerate(tqdm(samples, desc="Generating step-0 images")):
        image_path = images_dir / f"{sample.id}.png"

        # Skip if already generated
        if image_path.exists():
            logger.info(f"Skipping {sample.id} (already exists)")
            results_log.append({"id": sample.id, "status": "skipped"})
            continue

        try:
            result = generator.generate(sample.prompt)
            save_image(result.image, image_path)
            results_log.append({
                "id": sample.id,
                "prompt": sample.prompt,
                "dimension": sample.dimension,
                "image_path": str(image_path),
                "status": "success",
            })
            logger.info(f"[{i+1}/{len(samples)}] Generated {sample.id}")
        except Exception as e:
            logger.error(f"[{i+1}/{len(samples)}] Failed {sample.id}: {e}")
            results_log.append({"id": sample.id, "status": "failed", "error": str(e)})

    # Save generation log
    save_json(results_log, output_dir / "generation_log.json")
    logger.info(f"Generation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
