"""Quantitative evaluation.

The project has no ground-truth labels, so quality is reported with
reference-free image statistics plus a before/after histogram figure.
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import to_grayscale


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian - higher means more high-frequency detail."""
    gray = to_grayscale(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def rms_contrast(image: np.ndarray) -> float:
    """Root-mean-square contrast: standard deviation of intensity."""
    gray = to_grayscale(image).astype(np.float32)
    return float(gray.std())


def entropy(image: np.ndarray) -> float:
    """Shannon entropy of the intensity histogram, in bits.

    Rises when equalisation spreads pixels over more of the tonal range.
    """
    gray = to_grayscale(image)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def report(image: np.ndarray) -> dict[str, float]:
    """All metrics for one image."""
    return {
        "sharpness": round(sharpness(image), 2),
        "rms_contrast": round(rms_contrast(image), 2),
        "entropy": round(entropy(image), 3),
    }


def histogram_figure(before: np.ndarray, after: np.ndarray,
                     path: str) -> str:
    """Save a 2x2 figure: both images and their intensity histograms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b, a = to_grayscale(before), to_grayscale(after)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].imshow(b, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Input")
    axes[0, 1].imshow(a, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Output")
    for ax in (axes[0, 0], axes[0, 1]):
        ax.axis("off")

    axes[1, 0].hist(b.ravel(), bins=256, range=(0, 256), color="#444")
    axes[1, 0].set_title("Input histogram")
    axes[1, 1].hist(a.ravel(), bins=256, range=(0, 256), color="#444")
    axes[1, 1].set_title("Output histogram")
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xlabel("intensity")
        ax.set_ylabel("pixels")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
