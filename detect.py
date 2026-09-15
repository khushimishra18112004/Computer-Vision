"""Stage 2 - Page boundary detection.

Canny edges -> morphological closing -> contour extraction ->
quadrilateral approximation. Falls back to the minimum-area rotated
rectangle when the page border is broken by clutter.
"""

from __future__ import annotations

import cv2
import numpy as np


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny edge detection with thresholds derived from image statistics.

    Hard-coded thresholds fail across different lighting, so both
    hysteresis thresholds are set relative to the median intensity.
    """
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


def close_gaps(edges: np.ndarray, kernel_size: int = 5,
               iterations: int = 2) -> np.ndarray:
    """Morphological closing to bridge small breaks in the page outline."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left.

    Sum of coordinates is smallest at top-left and largest at bottom-right;
    the difference (y - x) separates the other two.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def find_document_contour(edges: np.ndarray,
                          min_area_ratio: float = 0.15) -> np.ndarray | None:
    """Return the 4 ordered corners of the largest page-like quadrilateral.

    Contours are ranked by area; each is approximated with Douglas-Peucker
    and accepted if it reduces to 4 convex points covering enough of the
    frame. If nothing qualifies, the largest contour's minAreaRect is used.
    """
    image_area = float(edges.shape[0] * edges.shape[1])
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_ratio * image_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx)

    largest = contours[0]
    if cv2.contourArea(largest) < min_area_ratio * image_area:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(largest))
    return order_corners(np.array(box))


def draw_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Overlay the detected quadrilateral for the visual report."""
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], True, (0, 255, 0), 3)
    for i, (x, y) in enumerate(corners.astype(int)):
        cv2.circle(canvas, (x, y), 9, (0, 0, 255), -1)
        cv2.putText(canvas, str(i), (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return canvas
