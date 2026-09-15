#!/usr/bin/env python3
"""DocuVision command-line interface.

Examples
--------
    python main.py samples/receipt.jpg
    python main.py samples/ --mode color --outdir outputs/
    python main.py samples/note.jpg --contrast hist --no-stages
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from docuvision import pipeline

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def collect_inputs(path: str) -> list[str]:
    """Expand a file, directory or glob into a sorted list of image paths."""
    if os.path.isdir(path):
        files: list[str] = []
        for ext in IMAGE_EXTENSIONS:
            files += glob.glob(os.path.join(path, f"*{ext}"))
            files += glob.glob(os.path.join(path, f"*{ext.upper()}"))
        return sorted(set(files))
    if any(ch in path for ch in "*?["):
        return sorted(glob.glob(path))
    return [path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docuvision",
        description="Scan and enhance document photos using classical CV.",
    )
    parser.add_argument("input", help="image file, directory or glob pattern")
    parser.add_argument("-o", "--outdir", default="outputs",
                        help="directory for results (default: outputs)")
    parser.add_argument("-m", "--mode", default="bw",
                        choices=["bw", "gray", "color"],
                        help="output style (default: bw)")
    parser.add_argument("-c", "--contrast", default="clahe",
                        choices=["clahe", "hist", "none"],
                        help="contrast normalisation used before detection")
    parser.add_argument("--no-stages", action="store_true",
                        help="save only the final scan, not intermediates")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    paths = collect_inputs(args.input)
    if not paths:
        print(f"no images found at: {args.input}", file=sys.stderr)
        return 1

    failures = 0
    for path in paths:
        try:
            result = pipeline.scan_file(path, mode=args.mode,
                                        contrast=args.contrast)
        except Exception as exc:                      # noqa: BLE001
            print(f"[FAIL] {path}: {exc}", file=sys.stderr)
            failures += 1
            continue

        written = pipeline.save_outputs(result, args.outdir,
                                        stages=not args.no_stages)
        status = "page detected" if result.corners_found else "no page - full frame"
        before, after = result.stats["input"], result.stats["output"]

        print(f"[ OK ] {os.path.basename(path)}  ({status})")
        print(f"       sharpness {before['sharpness']} -> {after['sharpness']}")
        print(f"       contrast  {before['rms_contrast']} -> {after['rms_contrast']}")
        print(f"       entropy   {before['entropy']} -> {after['entropy']}")
        print(f"       saved     {written['scanned']}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
