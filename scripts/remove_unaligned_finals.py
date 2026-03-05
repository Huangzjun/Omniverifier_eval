#!/usr/bin/env python3
"""Remove final images whose last TTS step is still is_aligned=False.

This aligns with the original OmniVerifier paper's logic where
step_final.png is only saved when verification passes.

Usage:
    # Dry run (default) — only prints what would be deleted
    python scripts/remove_unaligned_finals.py --tts_dir results/table3/t2i_reasonbench/cond8_OmniVerifier-TTS_Qwen-Image

    # Actually delete
    python scripts/remove_unaligned_finals.py --tts_dir results/table3/t2i_reasonbench/cond8_OmniVerifier-TTS_Qwen-Image --delete

    # Process multiple directories at once
    python scripts/remove_unaligned_finals.py \
        --tts_dir results/table3/t2i_reasonbench/cond8_OmniVerifier-TTS_Qwen-Image \
                  results/table3/t2i_reasonbench/cond8_OmniVerifier-TTS_Qwen-Image\(wanx-v1\) \
        --delete
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def find_last_step_meta(step_dir: Path) -> Path | None:
    """Find the highest-numbered step_N_meta.json in a sample's step dir."""
    metas = sorted(step_dir.glob("step_*_meta.json"))
    return metas[-1] if metas else None


def process_tts_dir(tts_dir: Path, delete: bool) -> dict:
    steps_dir = tts_dir / "steps"
    final_dir = tts_dir / "final"

    if not steps_dir.exists():
        print(f"  [SKIP] steps dir not found: {steps_dir}")
        return {}
    if not final_dir.exists():
        print(f"  [SKIP] final dir not found: {final_dir}")
        return {}

    stats = {"total": 0, "aligned": 0, "unaligned": 0, "deleted": 0, "no_final": 0}

    for sample_dir in sorted(steps_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        sample_id = sample_dir.name
        stats["total"] += 1

        last_meta = find_last_step_meta(sample_dir)
        if last_meta is None:
            continue

        with open(last_meta) as f:
            meta = json.load(f)

        is_aligned = meta.get("is_aligned", False)
        last_step = meta.get("step", "?")

        if is_aligned:
            stats["aligned"] += 1
            continue

        stats["unaligned"] += 1
        final_path = final_dir / f"{sample_id}.png"

        if not final_path.exists():
            stats["no_final"] += 1
            continue

        if delete:
            os.remove(final_path)
            stats["deleted"] += 1
            print(f"  [DELETED] {final_path.name}  (last step={last_step}, is_aligned=False)")
        else:
            print(f"  [WOULD DELETE] {final_path.name}  (last step={last_step}, is_aligned=False)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Remove final images whose last TTS step is still is_aligned=False"
    )
    parser.add_argument(
        "--tts_dir", type=str, nargs="+", required=True,
        help="Path(s) to TTS output directory (containing steps/ and final/ subdirs)",
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Actually delete files. Without this flag, only prints what would be deleted (dry run).",
    )
    args = parser.parse_args()

    if not args.delete:
        print("=" * 60)
        print("  DRY RUN — pass --delete to actually remove files")
        print("=" * 60)

    for tts_path in args.tts_dir:
        tts_dir = Path(tts_path)
        print(f"\nProcessing: {tts_dir}")
        stats = process_tts_dir(tts_dir, args.delete)

        if stats:
            print(f"\n  Summary for {tts_dir.name}:")
            print(f"    Total samples:     {stats['total']}")
            print(f"    Aligned (kept):    {stats['aligned']}")
            print(f"    Unaligned:         {stats['unaligned']}")
            if args.delete:
                print(f"    Deleted:           {stats['deleted']}")
            else:
                print(f"    Would delete:      {stats['unaligned'] - stats['no_final']}")
            print(f"    Already missing:   {stats['no_final']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
