"""Download DeepSeek-R1-0528 (FP8, 685B) from HuggingFace.

This is the latest R1 release (May 2025 update) with improved reasoning
and reduced hallucination. 163 safetensor shards, ~689 GB total.

No HuggingFace token required — the repo is public.

Usage:
    python download_model.py [--output-dir /path/to/store]
    python download_model.py --dry-run    # show what would be downloaded
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "deepseek-ai/DeepSeek-R1-0528"
DEFAULT_OUTPUT_DIR = Path.home() / "models" / "DeepSeek-R1-0528"


def show_repo_info(api: HfApi):
    info = api.model_info(REPO_ID, files_metadata=True)
    safetensor_files = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    total_bytes = sum(s.size or 0 for s in safetensor_files)
    other_files = [s for s in info.siblings if not s.rfilename.endswith(".safetensors") and not s.rfilename.startswith(".")]

    print(f"Repo:             {REPO_ID}")
    print(f"Safetensor shards: {len(safetensor_files)}")
    print(f"Safetensor total:  {total_bytes / 1e9:.1f} GB")
    print(f"Other files:       {len(other_files)}")
    for f in sorted(other_files, key=lambda x: x.rfilename):
        sz = f"({f.size / 1e6:.1f} MB)" if f.size and f.size > 1e6 else ""
        print(f"  {f.rfilename} {sz}")


def download(output_dir: Path, max_workers: int):
    print(f"Downloading {REPO_ID} to {output_dir}")
    print(f"This is ~689 GB — make sure you have enough disk space.\n")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        max_workers=max_workers,
    )
    print(f"\nDone. Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download DeepSeek-R1-0528 (FP8) from HuggingFace")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Where to save (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel download threads (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="Show repo info without downloading")
    args = parser.parse_args()

    api = HfApi()

    if args.dry_run:
        show_repo_info(api)
        return

    download(args.output_dir, args.max_workers)


if __name__ == "__main__":
    main()
