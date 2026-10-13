"""Stage 5 (optional, v1): fallback for needs_review results.

Order of effort per design doc section 4.5:
  (a) re-run extraction with the larger LLM
  (b) re-OCR with a second engine and merge
  (c) pass the image to a vision-language model (slow on CPU — failures only)

Re-validates the result; if it still fails, the caller routes to manual review
with the best attempt and failed checks attached.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import Config
from .extract import extract
from .ocr import run_ocr
from .validate import ValidationResult, validate


def run_fallback(
    image: np.ndarray,
    ocr_text: str,
    cfg: Config = Config(),
) -> ValidationResult:
    # (a) bigger LLM, same OCR text
    result = _try_larger_llm(ocr_text, ocr_text, cfg)
    if result.status != "needs_review":
        return result

    # (b) second OCR engine, re-extract, merge the better-covered fields
    result = _try_second_ocr(image, cfg)
    if result.status != "needs_review":
        return result

    # (c) VLM on the image directly — last resort, slow on CPU
    return _try_vlm(image, cfg)


def _try_larger_llm(layout_text: str, ocr_text: str, cfg: Config) -> ValidationResult:
    raw = extract(layout_text, cfg, model_path=cfg.fallback_llm_model_path)
    return validate(raw, ocr_text, cfg)


def _try_second_ocr(image: np.ndarray, cfg: Config) -> ValidationResult:
    alt_engine = "tesseract" if cfg.ocr_engine == "rapidocr" else "rapidocr"
    alt_cfg = replace(cfg, ocr_engine=alt_engine)
    ocr_result = run_ocr(image, alt_cfg)
    raw = extract(ocr_result.layout_text, cfg, model_path=cfg.fallback_llm_model_path)
    return validate(raw, ocr_result.layout_text, cfg)


def _try_vlm(image: np.ndarray, cfg: Config) -> ValidationResult:
    """Stub: wire up a real VLM call (e.g. Qwen2.5-VL via llama.cpp's multimodal
    support, or an API) here. Left unimplemented in v1 since it's explicitly
    the slowest, last-resort path per the design doc — build once (a) and (b)
    are measured and still leaving too much in needs_review."""
    raise NotImplementedError(
        "VLM fallback not wired up yet — (a) and (b) results should decide if this is even worth building"
    )
