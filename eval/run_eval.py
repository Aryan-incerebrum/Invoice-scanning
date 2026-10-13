#!/usr/bin/env python3
"""Scores a pipeline config against the golden set. Usage:

    python eval/run_eval.py --golden eval/golden --config configs/baseline.yaml

Golden set layout expected:
    eval/golden/
        invoice_001.jpg
        invoice_001.expected.json
        invoice_002.jpg
        invoice_002.expected.json
        ...

Computes exactly the metrics listed in design doc section 7:
header field exact-match rate (per field), line-item row match rate,
review rate, and the silent error rate (R6) — the headline number.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice2json.config import Config
from invoice2json.pipeline import process_invoice

HEADER_FIELDS = ["vendor_name", "invoice_number", "invoice_date", "currency", "total"]


def _numbers_equal(a, b, tol=0.01) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def _field_equal(field: str, expected, actual) -> bool:
    if expected is None and actual is None:
        return True
    if field in ("total", "subtotal", "tax_total", "discount_total"):
        return _numbers_equal(expected, actual)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return expected == actual


def _line_items_match(expected_items: list[dict], actual_items: list[dict]) -> tuple[int, int]:
    """Returns (matched_rows, total_expected_rows). A row matches if every
    field matches within tolerance — order-independent, greedy match."""
    remaining = list(actual_items)
    matched = 0
    for exp in expected_items:
        for i, act in enumerate(remaining):
            if all(_field_equal(f, exp.get(f), act.get(f)) for f in ("description", "quantity", "unit_price", "amount")):
                matched += 1
                remaining.pop(i)
                break
    return matched, len(expected_items)


def run_eval(golden_dir: Path, cfg: Config) -> dict:
    image_files = sorted(
        p for p in golden_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic", ".pdf")
    )

    per_field_hits = {f: 0 for f in HEADER_FIELDS}
    per_field_total = {f: 0 for f in HEADER_FIELDS}
    line_item_hits, line_item_total = 0, 0
    review_count = 0
    silent_error_count = 0  # ok/ok_with_warnings that actually contain a wrong field
    seconds = []

    per_invoice_rows = []

    for img_path in image_files:
        expected_path = img_path.with_suffix("").with_suffix(".expected.json")
        if not expected_path.exists():
            print(f"skip {img_path.name}: no expected JSON", file=sys.stderr)
            continue
        expected = json.loads(expected_path.read_text())

        t0 = time.time()
        result = process_invoice(img_path, cfg)
        seconds.append(time.time() - t0)

        actual = result.get("invoice") or {}
        status = result["status"]

        if status == "needs_review":
            review_count += 1

        row_wrong = False
        for f in HEADER_FIELDS:
            per_field_total[f] += 1
            if _field_equal(f, expected.get(f), actual.get(f)):
                per_field_hits[f] += 1
            else:
                row_wrong = True

        matched, total = _line_items_match(expected.get("line_items", []), actual.get("line_items", []))
        line_item_hits += matched
        line_item_total += total
        if total and matched < total:
            row_wrong = True

        if status in ("ok", "ok_with_warnings") and row_wrong:
            silent_error_count += 1

        per_invoice_rows.append({
            "file": img_path.name, "status": status,
            "header_correct": not row_wrong, "seconds": round(seconds[-1], 1),
        })

    n = len(per_invoice_rows)
    report = {
        "n_invoices": n,
        "header_field_accuracy": {
            f: round(per_field_hits[f] / per_field_total[f], 3) if per_field_total[f] else None
            for f in HEADER_FIELDS
        },
        "line_item_row_match_rate": round(line_item_hits / line_item_total, 3) if line_item_total else None,
        "review_rate": round(review_count / n, 3) if n else None,
        "silent_error_rate": round(silent_error_count / n, 3) if n else None,
        "seconds_median": round(statistics.median(seconds), 1) if seconds else None,
        "seconds_p95": round(statistics.quantiles(seconds, n=20)[18], 1) if len(seconds) >= 20 else (
            round(max(seconds), 1) if seconds else None
        ),
        "per_invoice": per_invoice_rows,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=Path(__file__).parent / "golden")
    parser.add_argument("--ocr-engine", default="rapidocr")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "report.md")
    args = parser.parse_args()

    cfg = Config(ocr_engine=args.ocr_engine)
    
    if args.llm_model:
        cfg.llm_model = args.llm_model
    report = run_eval(args.golden, cfg)

    lines = [
        f"# Eval report\n",
        f"- Invoices: {report['n_invoices']}",
        f"- OCR engine: {cfg.ocr_engine}",
        f"- LLM: {cfg.llm_model}",
        f"- Review rate: {report['review_rate']}",
        f"- Silent error rate (R6, headline metric): {report['silent_error_rate']}",
        f"- Line-item row match rate: {report['line_item_row_match_rate']}",
        f"- Median seconds/invoice: {report['seconds_median']}  (p95: {report['seconds_p95']})",
        f"\n## Header field accuracy\n",
    ]
    for f, acc in report["header_field_accuracy"].items():
        lines.append(f"- {f}: {acc}")
    lines.append("\n## Per-invoice\n")
    for row in report["per_invoice"]:
        lines.append(f"- {row['file']}: {row['status']}, header_correct={row['header_correct']}, {row['seconds']}s")

    args.out.write_text("\n".join(lines))
    print(f"Wrote {args.out}")
    print(json.dumps(report, indent=2))
