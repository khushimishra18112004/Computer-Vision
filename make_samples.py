#!/usr/bin/env python3
"""Generate synthetic test images.

Renders a clean page, warps it with a random homography, then adds an
uneven lighting gradient and Gaussian noise. This gives the pipeline
realistic inputs without shipping copyrighted photographs in the repo.

Usage:  python make_samples.py [--outdir samples] [--count 3]
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

PAGE_W, PAGE_H = 850, 1100
CANVAS_W, CANVAS_H = 1280, 1600

LINES = [
    ("INVOICE / LAB RECORD", 1.3, 3),
    ("", 0, 0),
    ("Name      : Khushi Mishra", 0.8, 2),
    ("Course    : Computer Vision", 0.8, 2),
    ("Date      : 15-09-2026", 0.8, 2),
    ("", 0, 0),
    ("Item              Qty      Amount", 0.8, 2),
    ("--------------------------------", 0.7, 1),
    ("Edge detection      1        120", 0.75, 2),
    ("Perspective warp    2        340", 0.75, 2),
    ("Histogram eq.       1         90", 0.75, 2),
    ("Adaptive thresh.    3        275", 0.75, 2),
    ("--------------------------------", 0.7, 1),
    ("TOTAL                        825", 0.85, 2),
]


def render_page() -> np.ndarray:
    """Draw a flat, well-lit document."""
    page = np.full((PAGE_H, PAGE_W, 3), 245, dtype=np.uint8)
    cv2.rectangle(page, (40, 40), (PAGE_W - 40, PAGE_H - 40), (205, 205, 205), 2)

    y = 140
    for text, scale, thickness in LINES:
        if text:
            cv2.putText(page, text, (80, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (25, 25, 25), thickness, cv2.LINE_AA)
        y += 55 if text else 25

    # a block of body text so sharpness metrics have fine detail to measure
    for i in range(10):
        cv2.putText(page, "the quick brown fox jumps over the lazy dog 0123456789",
                    (80, y + i * 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (60, 60, 60), 1, cv2.LINE_AA)
    return page


def place_on_desk(page: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Composite the page onto a textured background under a random homography."""
    desk = np.full((CANVAS_H, CANVAS_W, 3), 95, dtype=np.uint8)
    desk += rng.integers(-18, 18, desk.shape, dtype=np.int16).astype(np.uint8)

    h, w = page.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    margin_x, margin_y = 150, 190
    jitter = rng.integers(-110, 110, (4, 2)).astype(np.float32)
    dst = np.float32([
        [margin_x, margin_y],
        [CANVAS_W - margin_x, margin_y],
        [CANVAS_W - margin_x, CANVAS_H - margin_y],
        [margin_x, CANVAS_H - margin_y],
    ]) + jitter

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, matrix, (CANVAS_W, CANVAS_H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix,
                               (CANVAS_W, CANVAS_H))
    desk[mask > 0] = warped[mask > 0]
    return desk


def degrade(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a lighting gradient plus sensor noise."""
    h, w = image.shape[:2]
    cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    falloff = 1.0 - 0.60 * (dist / dist.max())
    lit = image.astype(np.float32) * falloff[..., None]

    noise = rng.normal(0, 7, lit.shape)
    return np.clip(lit + noise, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic samples.")
    parser.add_argument("--outdir", default="samples")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    page = render_page()

    for i in range(1, args.count + 1):
        image = degrade(place_on_desk(page, rng), rng)
        path = os.path.join(args.outdir, f"sample_{i:02d}.jpg")
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
