"""Unit tests. Run with:  pytest -q"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docuvision import detect, enhance, metrics, pipeline, preprocess, transform  # noqa: E402


@pytest.fixture
def synthetic_page() -> np.ndarray:
    """A dark canvas with a bright, slightly rotated quadrilateral page."""
    canvas = np.full((600, 800, 3), 40, dtype=np.uint8)
    quad = np.array([[150, 90], [660, 130], [630, 520], [180, 480]], dtype=np.int32)
    cv2.fillPoly(canvas, [quad], (235, 235, 235))
    cv2.putText(canvas, "TEST", (260, 320), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (20, 20, 20), 4)
    return canvas


def test_grayscale_is_single_channel(synthetic_page):
    assert preprocess.to_grayscale(synthetic_page).ndim == 2


def test_equalization_widens_tonal_range():
    """A low-contrast image should gain contrast after equalisation."""
    flat = np.random.default_rng(0).integers(100, 140, (256, 256), dtype=np.uint8)
    equalized = preprocess.histogram_equalization(flat)
    assert equalized.std() > flat.std()


def test_clahe_preserves_shape_and_dtype(synthetic_page):
    gray = preprocess.to_grayscale(synthetic_page)
    out = preprocess.clahe(gray)
    assert out.shape == gray.shape and out.dtype == np.uint8


def test_corner_ordering_is_clockwise_from_top_left():
    unordered = np.array([[10, 90], [90, 10], [10, 10], [90, 90]], dtype=np.float32)
    tl, tr, br, bl = detect.order_corners(unordered)
    assert tuple(tl) == (10, 10)
    assert tuple(tr) == (90, 10)
    assert tuple(br) == (90, 90)
    assert tuple(bl) == (10, 90)


def test_detects_four_corners(synthetic_page):
    gray = preprocess.preprocess(synthetic_page)
    edges = detect.close_gaps(detect.auto_canny(gray))
    corners = detect.find_document_contour(edges)
    assert corners is not None and corners.shape == (4, 2)


def test_warp_produces_axis_aligned_rectangle(synthetic_page):
    corners = np.array([[150, 90], [660, 130], [630, 520], [180, 480]],
                       dtype=np.float32)
    warped = transform.four_point_transform(synthetic_page, corners)
    assert warped.shape[0] > 0 and warped.shape[1] > 0
    # the bright page should now fill the frame
    assert preprocess.to_grayscale(warped).mean() > 150


def test_resize_returns_usable_ratio(synthetic_page):
    small, ratio = transform.resize_to_height(synthetic_page, 300)
    assert small.shape[0] == 300
    assert 0 < ratio < 1


def test_bw_output_is_binary(synthetic_page):
    out = enhance.enhance(synthetic_page, mode="bw")
    assert set(np.unique(out)).issubset({0, 255})


@pytest.mark.parametrize("mode", ["bw", "gray", "color"])
def test_all_modes_run(synthetic_page, mode):
    assert enhance.enhance(synthetic_page, mode=mode).size > 0


def test_unknown_mode_raises(synthetic_page):
    with pytest.raises(ValueError):
        enhance.enhance(synthetic_page, mode="sepia")


def test_metrics_are_finite(synthetic_page):
    stats = metrics.report(synthetic_page)
    assert set(stats) == {"sharpness", "rms_contrast", "entropy"}
    assert all(np.isfinite(v) for v in stats.values())


def test_pipeline_end_to_end(synthetic_page):
    result = pipeline.scan(synthetic_page, source="unit_test.png")
    assert result.corners_found
    assert result.scanned.size > 0
    assert "input" in result.stats and "output" in result.stats


def test_pipeline_falls_back_without_a_page():
    """A featureless frame has no page boundary; the run must not crash."""
    blank = np.full((400, 400, 3), 128, dtype=np.uint8)
    result = pipeline.scan(blank)
    assert result.scanned.shape[:2] == blank.shape[:2]


def test_save_outputs_writes_files(synthetic_page, tmp_path):
    result = pipeline.scan(synthetic_page, source="doc.png")
    written = pipeline.save_outputs(result, str(tmp_path))
    assert os.path.exists(written["scanned"])
    assert os.path.exists(written["02_edges"])


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        pipeline.scan_file("no_such_image.jpg")
