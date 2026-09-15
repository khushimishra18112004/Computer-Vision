"""End-to-end scanning pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import detect, enhance, metrics, preprocess, transform


@dataclass
class ScanResult:
    """Everything produced for a single input image."""
    source: str
    original: np.ndarray
    processed: np.ndarray
    edges: np.ndarray
    outlined: np.ndarray
    warped: np.ndarray
    scanned: np.ndarray
    corners_found: bool
    stats: dict = field(default_factory=dict)


def scan(image: np.ndarray, source: str = "<array>", mode: str = "bw",
         contrast: str = "clahe") -> ScanResult:
    """Run preprocess -> detect -> warp -> enhance on one image."""
    small, ratio = transform.resize_to_height(image, 900)

    processed = preprocess.preprocess(small, contrast=contrast)
    edges = detect.close_gaps(detect.auto_canny(processed))
    corners = detect.find_document_contour(edges)

    if corners is None:
        # No page boundary found - enhance the full frame rather than fail.
        corners_found = False
        outlined = small if small.ndim == 3 else cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        outlined = outlined.copy()
        warped = image.copy()
    else:
        corners_found = True
        outlined = detect.draw_corners(small, corners)
        corners = corners / ratio  # map back to full resolution
        warped = transform.four_point_transform(image, corners)

    scanned = enhance.enhance(warped, mode=mode)

    return ScanResult(
        source=source,
        original=image,
        processed=processed,
        edges=edges,
        outlined=outlined,
        warped=warped,
        scanned=scanned,
        corners_found=corners_found,
        stats={"input": metrics.report(image), "output": metrics.report(scanned)},
    )


def scan_file(path: str, mode: str = "bw", contrast: str = "clahe") -> ScanResult:
    """Load an image from disk and scan it."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return scan(image, source=path, mode=mode, contrast=contrast)


def save_outputs(result: ScanResult, outdir: str,
                 stages: bool = True) -> dict[str, str]:
    """Write the scan (and optionally every intermediate stage) to `outdir`."""
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(result.source))[0] or "scan"

    written: dict[str, str] = {}

    def _write(tag: str, image: np.ndarray) -> None:
        path = os.path.join(outdir, f"{stem}_{tag}.png")
        cv2.imwrite(path, image)
        written[tag] = path

    _write("scanned", result.scanned)
    if stages:
        _write("01_preprocessed", result.processed)
        _write("02_edges", result.edges)
        _write("03_detected", result.outlined)
        _write("04_warped", result.warped)
        metrics.histogram_figure(
            result.original, result.scanned,
            os.path.join(outdir, f"{stem}_05_histograms.png"),
        )
        written["histograms"] = os.path.join(outdir, f"{stem}_05_histograms.png")

    return written
