"""Orchestrator. One public function: process_invoice(source_path, cfg) -> dict.

Implements the flow in design doc section 3:
  preprocess -> OCR -> extract -> validate -> (fallback if needed) -> output
Nothing is written as JSON until the validation gate passes or the fallback
chain is exhausted. Every run also writes a debug bundle.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import cv2

from .config import Config
from .ocr import run_ocr
from .extract import extract
from .preprocess import preprocess
from .validate import validate

SCHEMA_VERSION = "1.0"
PIPELINE_VERSION = "0.1.0"


def _write_debug_bundle(debug_dir: Path, stem: str, **artifacts):
    out_dir = debug_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        if content is None:
            continue
        if name.endswith(".jpg") and hasattr(content, "shape"):
            cv2.imwrite(str(out_dir / name), content)
        else:
            (out_dir / name).write_text(
                content if isinstance(content, str) else json.dumps(content, indent=2, default=str), 
                encoding ="utf-8"
            )


def process_invoice(source_path: str | Path, cfg: Config = Config()) -> dict:
    source_path = Path(source_path)
    stem = source_path.stem
    started = time.time()

    base_output = {
        "schema_version": SCHEMA_VERSION,
        "source_file": source_path.name,
        "meta": {
            "pipeline_version": PIPELINE_VERSION,
            "ocr_engine": cfg.ocr_engine,
            "llm": Path(cfg.llm_model).stem,
        },
    }

    # --- Stage 1: preprocess ---
    pre = preprocess(source_path, cfg)
    if not pre.ok:
        output = {
            **base_output,
            "status": "rejected_low_quality",
            "invoice": None,
            "checks": {},
            "warnings": [pre.reject_reason],
        }
        _write_debug_bundle(cfg.debug_dir, stem, **{"preprocess_result.json": {
            "sharpness": pre.sharpness, "brightness": pre.brightness,
            "page_area_ratio": pre.page_area_ratio, "reject_reason": pre.reject_reason,
        }})
        return output

    try:
        # --- Stage 2: OCR ---
        ocr_result = run_ocr(pre.image, cfg)

        print(stem, "OCR length:", len(ocr_result.layout_text))
        try:
            _write_debug_bundle(
                cfg.debug_dir,
                stem,
                **{"ocr_text.txt": ocr_result.layout_text},
            )
        except Exception as e:
            print(f"OCR SAVE FAILED for {stem}: {e}")
            raise
        print("SAVED OCR:", cfg.debug_dir / stem / "ocr_text.txt")

        # --- Stage 3: extract ---
        raw_invoice = extract(ocr_result.layout_text, cfg)
        _write_debug_bundle(
            cfg.debug_dir,
            stem,
            **{
                "llm_output.json": raw_invoice,
            },
        )
        # --- Stage 4: validate ---
        result = validate(raw_invoice, ocr_result.layout_text, cfg)

        # --- Stage 5: fallback if needed ---
        if result.status == "needs_review" and cfg.fallback_enabled:
            from .fallback import run_fallback
            try:
                result = run_fallback(pre.image, ocr_result.layout_text, cfg)
            except NotImplementedError:
                pass  # (a)/(b) already tried inside run_fallback; VLM stub not built yet

        elapsed = time.time() - started
        output = {
            **base_output,
            "status": result.status,
            "invoice": result.invoice,
            "checks": result.checks,
            "warnings": result.warnings,
        }
        output["meta"]["seconds"] = round(elapsed, 1)

        _write_debug_bundle(
            cfg.debug_dir, stem,
            **{
                "preprocessed.jpg": pre.image,
                "ocr_layout_text.txt": ocr_result.layout_text,
                "raw_model_output.json": raw_invoice,
                "validation_report.json": {
                    "status": result.status, "checks": result.checks,
                    "warnings": result.warnings, "failed_hard_checks": result.failed_hard_checks,
                },
            },
        )
        return output

    except Exception as exc:  # never crash the batch on one bad invoice
        output = {
            **base_output,
            "status": "needs_review",
            "invoice": None,
            "checks": {},
            "warnings": [f"pipeline error: {exc}"],
        }
        _write_debug_bundle(cfg.debug_dir, stem, **{"traceback.txt": traceback.format_exc()})
        return output


def write_result(output: dict, cfg: Config = Config()) -> Path:
    stem = Path(output["source_file"]).stem
    target_dir = cfg.review_dir if output["status"] == "needs_review" else cfg.done_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{stem}.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    return out_path
