"""Stage 1 - Preprocessing.

Grayscale conversion, noise suppression and contrast normalisation.
Contrast normalisation is what makes corner detection survive bad lighting,
so both global histogram equalisation and CLAHE are implemented here.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Suppress sensor noise while keeping page edges intact.

    gaussian  - fast, blurs edges slightly
    median    - best against salt-and-pepper noise
    bilateral - edge preserving, default because we need the page border sharp
    """
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (5, 5), 0)
    if method == "median":
        return cv2.medianBlur(gray, 5)
    if method == "bilateral":
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    raise ValueError(f"unknown denoise method: {method}")


def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """Global histogram equalisation.

    Redistributes intensities across the full 0-255 range using the
    cumulative distribution function. Boosts global contrast but amplifies
    noise in already-bright regions - see clahe() for the local variant.
    """
    return cv2.equalizeHist(gray)


def clahe(gray: np.ndarray, clip_limit: float = 2.5,
          tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation.

    Equalises each tile independently and clips the histogram before
    building the CDF, which prevents the noise amplification that plain
    equalizeHist produces on document photos with uneven lighting.
    """
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return op.apply(gray)


def normalize_illumination(gray: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """Remove the low-frequency lighting gradient (shadow / lamp falloff).

    A heavy median blur estimates the background illumination; dividing the
    original by that estimate flattens the page to uniform brightness.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    background = cv2.medianBlur(gray, kernel_size)
    background = np.where(background == 0, 1, background).astype(np.float32)
    flat = (gray.astype(np.float32) / background) * 128.0
    return np.clip(flat, 0, 255).astype(np.uint8)


def preprocess(image: np.ndarray, contrast: str = "clahe",
               denoise_method: str = "bilateral") -> np.ndarray:
    """Full stage-1 chain: grayscale -> contrast -> denoise."""
    gray = to_grayscale(image)

    if contrast == "clahe":
        gray = clahe(gray)
    elif contrast == "hist":
        gray = histogram_equalization(gray)
    elif contrast == "none":
        pass
    else:
        raise ValueError(f"unknown contrast mode: {contrast}")

    return denoise(gray, denoise_method)
