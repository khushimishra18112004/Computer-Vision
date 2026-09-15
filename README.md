# DocuVision — Adaptive Document Scanner

A classical computer vision pipeline that turns a skewed, badly-lit phone photo of a
document into a flat, evenly-lit, high-contrast scan. No neural networks, no training
data, no GPU — every stage is a deterministic image processing operation.

**Course:** Computer Vision (Flipped Course — Evaluated Project)
**Author:** Khushi Mishra · 24BAI10509 · VIT Bhopal

---

## Pipeline

The system runs five stages in sequence: preprocessing, page boundary detection, perspective rectification, enhancement, and evaluation. Every intermediate stage is written to disk (see `--no-stages` in Usage below), so the transformation is inspectable step by step.

---

## The problem

A photo of a document is not a scan. Three things go wrong at once:

1. **Perspective distortion** — the camera is never perfectly parallel to the page, so
   the rectangle arrives as an arbitrary quadrilateral.
2. **Uneven illumination** — one corner is lit, another is in shadow, so no single
   brightness threshold works across the page.
3. **Noise and softness** — sensor noise and hand shake blur the text.

DocuVision fixes all three in one pass and outputs a page-only image in black-and-white,
grayscale or colour.

---

## Results

Three document layouts, each with a different random homography, lighting falloff and
noise level. Boundary detection succeeded on all three. Measured with `--mode bw`:

| Sample | Page found | Sharpness (in → out) | RMS contrast (in → out) | Entropy (in → out) |
|---|---|---|---|---|
| `invoice` | yes | 1420 → 12742 | 59.5 → 57.1 | 7.52 → 0.30 |
| `notes`   | yes | 1429 → 12889 | 71.3 → 50.4 | 7.07 → 0.25 |
| `form`    | yes | 1459 → 12031 | 70.9 → 53.8 | 7.31 → 0.27 |

Sharpness rises roughly 9× because thresholding replaces soft anti-aliased character
edges with hard binary transitions. Entropy collapses toward 0.3 bits **by design** — a
binary image occupies two intensity levels instead of 256, so this number reflects the
output format rather than a loss of information. In `--mode gray`, where the full tonal
range is kept, entropy stays near the input value while contrast improves.

The clearest evidence is in the before/after histogram (generated per run at `outputs/<name>_05_histograms.png`). The input distribution is broad and multi-modal — dark background, lit page and shadowed page form separate populations. After processing the background is cropped away entirely and the remaining pixels separate cleanly into ink and paper, which is exactly the separation an OCR engine needs.

---

## How each stage works

**1. Preprocessing.** Grayscale conversion, then contrast normalisation. Plain
`equalizeHist` stretches intensities globally using the histogram's CDF, but it amplifies
noise in already-bright regions. CLAHE equalises each 8×8 tile independently and clips
the histogram before building the CDF, which bounds the transfer slope and therefore
bounds noise amplification — this is what actually works on document photos carrying a
lighting gradient. A bilateral filter then removes noise while keeping the page border
sharp; a Gaussian blur would soften exactly the edge the next stage depends on.

**2. Detection.** Canny thresholds are derived from the image's median intensity rather
than hard-coded, so the same code handles bright and dim photos. Morphological closing
bridges small breaks in the outline. Contours are ranked by area and simplified with
Douglas-Peucker; the first convex 4-point result covering at least 15% of the frame is
accepted as the page. If no clean quadrilateral exists, the largest contour's
minimum-area rotated rectangle is used as a fallback, and if that also fails the full
frame is processed rather than erroring out.

**3. Rectification.** The four corners are ordered top-left → top-right → bottom-right →
bottom-left using coordinate sums and differences. A 3×3 homography maps them onto a
rectangle sized from the longer of each pair of opposing edges, so no content is
compressed. Detection runs on a 900px-tall copy for speed; corners are scaled back up
before warping so the output keeps full resolution.

**4. Enhancement.** A 51-pixel median blur estimates the background illumination —
large enough to erase all text while preserving the lighting gradient — and dividing by
it flattens the page. Adaptive Gaussian thresholding then computes a separate threshold
per 31×31 neighbourhood, which is what lets a shadowed corner binarise correctly where
Otsu's global threshold would render it solid black. An unsharp mask restores detail lost
to bicubic interpolation. Colour mode applies CLAHE to the L channel in LAB space only,
so contrast improves without shifting ink or stamp hues.

**5. Evaluation.** With no ground truth available, quality is reported with
reference-free statistics: variance of the Laplacian (sharpness), standard deviation of
intensity (RMS contrast), and Shannon entropy of the histogram.

---

## Install

```bash
git clone https://github.com/khushimishra18112004/Computer-Vision.git
cd docuvision
pip install -r requirements.txt
```

Requires Python 3.9+.

## Usage

Generate test images (the repo ships no photographs — samples are rendered):

```bash
python make_samples.py                      # one of each layout
python make_samples.py --layout notes --count 4
```

Scan them:

```bash
python main.py samples/invoice.jpg              # single file, B&W
python main.py samples/ --mode color            # whole folder, colour
python main.py samples/ --mode gray --no-stages # final scan only
```

Rebuild every figure in this README from a live run:

```bash
python make_figures.py
```

### Options

| Flag | Values | Default | Effect |
|---|---|---|---|
| `-m`, `--mode` | `bw`, `gray`, `color` | `bw` | Output style |
| `-c`, `--contrast` | `clahe`, `hist`, `none` | `clahe` | Contrast method before detection |
| `-o`, `--outdir` | path | `outputs` | Where results are written |
| `--no-stages` | flag | off | Skip saving intermediate images |

By default every stage is written out (`_01_preprocessed`, `_02_edges`, `_03_detected`,
`_04_warped`, `_05_histograms`), which makes the pipeline easy to inspect and debug.

### As a library

```python
from docuvision import scan_file, save_outputs

result = scan_file("photo.jpg", mode="gray")
print(result.corners_found, result.stats)
save_outputs(result, "outputs/")
```

---

## Project layout

```
docuvision/
├── docuvision/
│   ├── preprocess.py    grayscale, denoising, equalisation, CLAHE
│   ├── detect.py        auto-Canny, contours, corner detection & ordering
│   ├── transform.py     homography and perspective warp
│   ├── enhance.py       thresholding, sharpening, bw/gray/colour modes
│   ├── metrics.py       sharpness, contrast, entropy, histogram figure
│   └── pipeline.py      stage orchestration and output writing
├── tests/
│   └── test_pipeline.py 17 unit tests
├── main.py              command-line interface
├── make_samples.py      synthetic test image generator (3 layouts)
├── make_figures.py      rebuilds every README figure from a live run
├── samples/             generated inputs
└── assets/              figures used in this README
```

## Tests

```bash
pytest -q
```

17 tests covering corner ordering, contrast operations, detection, warping, all three
output modes, metric validity, the no-page fallback path, and file writing.

---

## Concepts from the course used here

Colour space conversion (BGR / grayscale / LAB) · histogram equalisation · CLAHE ·
Gaussian, median and bilateral filtering · Canny edge detection and hysteresis
thresholding · morphological closing · contour extraction and hierarchy · Douglas-Peucker
polygon approximation · convexity testing · minimum-area rotated rectangle · homography
estimation and perspective warping · bicubic interpolation · adaptive thresholding ·
unsharp masking · image quality metrics.

## Limitations

- Requires reasonable contrast between page and background; a white page on a white desk
  will not be detected.
- Assumes a single, mostly-flat document. Curved book pages are not dewarped.
- Residual skew after rectification is not corrected; a Hough-based text-line angle
  estimator would handle it.
- No text recognition — the output is a clean image, intended as the input to an OCR
  engine rather than a replacement for one.

## License

MIT — see [LICENSE](LICENSE).
