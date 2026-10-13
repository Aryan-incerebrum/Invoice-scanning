"""Stage 1: ingest and preprocess.

One public function: preprocess(source_path, cfg) -> PreprocessResult.
Accepts JPG/PNG/HEIC/PDF. Returns a cleaned grayscale-contrast-adjusted image
plus a quality verdict. Nothing downstream runs if the quality gate fails.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC input simply won't open; caller sees a clear error instead

from .config import Config


@dataclass
class PreprocessResult:
    ok: bool
    image: Optional[np.ndarray]   # BGR, uint8 — None if rejected
    sharpness: float
    brightness: float
    page_area_ratio: float
    reject_reason: Optional[str] = None


def _load_as_bgr(source_path: Path) -> np.ndarray:
    if source_path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(source_path))
        page = pdf[0]
        bitmap = page.render(scale=300 / 72)  # 300 DPI
        pil_img = bitmap.to_pil()
    else:
        pil_img = Image.open(source_path)
        pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _find_page_quad(img: np.ndarray) -> Optional[np.ndarray]:
    """Largest 4-point contour, assumed to be the page. None if not confident."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = h * w
    best = None
    best_area = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area and area > 0.2 * frame_area:
                best, best_area = approx, area

    if best is None:
        return None
    return best.reshape(4, 2).astype("float32")


def _order_corners(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def _warp(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = _order_corners(quad)
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width, height = max(width, 1), max(height, 1)

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    return cv2.warpPerspective(img, matrix, (width, height))


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess(source_path: str | Path, cfg: Config = Config()) -> PreprocessResult:
    source_path = Path(source_path)
    img = _load_as_bgr(source_path)
    orig_h, orig_w = img.shape[:2]
    frame_area = orig_h * orig_w

    quad = _find_page_quad(img)
    page_area_ratio = 1.0
    if quad is not None:
        page_area_ratio = float(cv2.contourArea(quad) / frame_area)
        img = _warp(img, quad)
    # if no quad found, proceed with the uncropped frame rather than failing —
    # a confident crop is a bonus, not a requirement, since cropping too
    # aggressively has repeatedly hurt downstream OCR/VLM accuracy

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = _sharpness(gray)
    brightness = float(gray.mean())

    reject_reason = None
    if sharpness < cfg.min_sharpness:
        reject_reason = f"image too blurry (sharpness={sharpness:.1f}, min={cfg.min_sharpness})"
    elif quad is not None and page_area_ratio < cfg.min_page_area_ratio:
        reject_reason = f"page too small in frame (area_ratio={page_area_ratio:.2f})"

    if reject_reason:
        return PreprocessResult(False, None, sharpness, brightness, page_area_ratio, reject_reason)

    # downscale long edge, then contrast-enhance — grayscale, not binarized,
    # per the design doc note that modern OCR does better on grayscale
    h, w = gray.shape
    long_edge = max(h, w)
    if long_edge > cfg.target_long_edge_px:
        scale = cfg.target_long_edge_px / long_edge
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = _clahe(gray)
    final = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return PreprocessResult(True, final, sharpness, brightness, page_area_ratio, None)
