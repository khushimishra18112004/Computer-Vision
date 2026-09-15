"""Stage 3 - Geometric rectification.

Maps the detected quadrilateral onto a rectangle using a 3x3 homography,
which removes the keystone distortion of a hand-held camera shot.
"""

from __future__ import annotations

import cv2
import numpy as np


def target_size(corners: np.ndarray) -> tuple[int, int]:
    """Estimate output width/height from the quadrilateral's edge lengths.

    The longer of each opposing pair is used so no content is compressed.
    """
    tl, tr, br, bl = corners

    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    return max(int(width), 1), max(int(height), 1)


def four_point_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the region bounded by `corners` into a front-facing rectangle."""
    width, height = target_size(corners)

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    return cv2.warpPerspective(image, matrix, (width, height),
                               flags=cv2.INTER_CUBIC)


def resize_to_height(image: np.ndarray, height: int = 900) -> tuple[np.ndarray, float]:
    """Downscale for detection speed. Returns the image and the scale factor.

    Detection runs on the small copy; corners are divided by this ratio to
    map back onto the full-resolution original before warping.
    """
    if image.shape[0] <= height:
        return image.copy(), 1.0
    ratio = height / float(image.shape[0])
    width = int(image.shape[1] * ratio)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), ratio
