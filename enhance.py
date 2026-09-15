"""Stage 4 - Output enhancement.

Turns the rectified crop into a readable scan. Three output modes are
offered because the right one depends on the source document.
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import clahe, normalize_illumination, to_grayscale


def unsharp_mask(image: np.ndarray, amount: float = 1.4,
                 radius: int = 5) -> np.ndarray:
    """Sharpen by subtracting a blurred copy from the original.

    Recovers the high-frequency detail lost to the bicubic warp.
    """
    if radius % 2 == 0:
        radius += 1
    blurred = cv2.GaussianBlur(image, (radius, radius), 0)
    sharp = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def scan_bw(warped: np.ndarray, block_size: int = 31, c: int = 12) -> np.ndarray:
    """High-contrast black-and-white scan via adaptive Gaussian thresholding.

    A single global threshold fails when one side of the page is shadowed,
    so the threshold is computed per neighbourhood instead.
    """
    gray = to_grayscale(warped)
    gray = normalize_illumination(gray)
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c,
    )


def scan_gray(warped: np.ndarray) -> np.ndarray:
    """Grayscale scan - keeps pencil marks and photographs readable."""
    gray = to_grayscale(warped)
    gray = normalize_illumination(gray)
    gray = clahe(gray, clip_limit=2.0)
    return unsharp_mask(gray)


def scan_color(warped: np.ndarray) -> np.ndarray:
    """Colour scan with contrast fixed in LAB space.

    CLAHE is applied to the L (lightness) channel only, so contrast
    improves without shifting the hues of stamps, ink or highlighter.
    """
    if warped.ndim == 2:
        return scan_gray(warped)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe(l, clip_limit=2.0)
    merged = cv2.merge((l, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return unsharp_mask(result, amount=0.8)


MODES = {"bw": scan_bw, "gray": scan_gray, "color": scan_color}


def enhance(warped: np.ndarray, mode: str = "bw") -> np.ndarray:
    """Apply the requested output mode."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}. choose from {sorted(MODES)}")
    return MODES[mode](warped)
