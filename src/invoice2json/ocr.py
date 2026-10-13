"""Stage 2: OCR. One public function: run_ocr(image, cfg) -> OcrResult.

Default engine is RapidOCR (PP-OCRv5 models, ONNX Runtime, CPU). Tesseract 5
is the benchmark alternative — see eval/run_eval.py for the bake-off that
decides between them on the golden set, not preference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class TextLine:
    text: str
    box: list[tuple[float, float]]  # 4 corner points
    confidence: float


@dataclass
class OcrResult:
    lines: list[TextLine]
    layout_text: str   # reading-order text with column gaps kept as tabs — this is the LLM's input


def _rows_from_boxes(lines: list[TextLine], row_tolerance_px: float = 12.0) -> list[list[TextLine]]:
    """Group lines into rows by vertical center, then sort each row left to right.

    Column gaps become tab characters so the LLM extracting fields can see
    table structure rather than a flat wall of text.
    """
    if not lines:
        return []

    def y_center(line: TextLine) -> float:
        return sum(p[1] for p in line.box) / 4.0

    def x_left(line: TextLine) -> float:
        return min(p[0] for p in line.box)

    sorted_lines = sorted(lines, key=y_center)
    rows: list[list[TextLine]] = []
    current_row: list[TextLine] = []
    current_y: float | None = None

    for line in sorted_lines:
        y = y_center(line)
        if current_y is None or abs(y - current_y) <= row_tolerance_px:
            current_row.append(line)
            current_y = y if current_y is None else (current_y + y) / 2
        else:
            rows.append(sorted(current_row, key=x_left))
            current_row = [line]
            current_y = y
    if current_row:
        rows.append(sorted(current_row, key=x_left))
    return rows


def _layout_text(lines: list[TextLine]) -> str:
    rows = _rows_from_boxes(lines)
    out_lines = []
    for row in rows:
        out_lines.append("\t".join(l.text for l in row))
    return "\n".join(out_lines)


def _run_rapidocr(image: np.ndarray) -> list[TextLine]:
    from rapidocr_onnxruntime import RapidOCR  # import here: optional heavy dep
    engine = RapidOCR()
    result, _ = engine(image)
    lines = []
    if result:
        for box, text, conf in result:
            lines.append(TextLine(text=text, box=[tuple(p) for p in box], confidence=float(conf)))
    return lines


def _run_tesseract(image: np.ndarray) -> list[TextLine]:
    import pytesseract
    from pytesseract import Output
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    lines = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        conf = float(data["conf"][i]) / 100.0 if data["conf"][i] != "-1" else 0.0
        box = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        lines.append(TextLine(text=text, box=box, confidence=conf))
    return lines


def run_ocr(image: np.ndarray, cfg: Config = Config()) -> OcrResult:
    if cfg.ocr_engine == "rapidocr":
        lines = _run_rapidocr(image)
    elif cfg.ocr_engine == "tesseract":
        lines = _run_tesseract(image)
    else:
        raise ValueError(f"Unknown ocr_engine: {cfg.ocr_engine}")

    return OcrResult(lines=lines, layout_text=_layout_text(lines))
