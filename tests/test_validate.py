import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice2json.config import Config
from invoice2json.validate import _to_number, _normalize_date, _normalize_currency, validate


def test_to_number_handles_thousands_separator():
    cfg = Config(thousands_separator=",", decimal_separator=".")
    assert _to_number("1,250.00", cfg) == 1250.00
    assert _to_number("₹2,384.00", cfg) == 2384.00
    assert _to_number(None, cfg) is None


def test_to_number_handles_european_format():
    cfg = Config(thousands_separator=".", decimal_separator=",")
    assert _to_number("1.250,00", cfg) == 1250.00


def test_normalize_date_iso_passthrough():
    cfg = Config()
    iso, ambiguous = _normalize_date("2026-09-28", cfg)
    assert iso == "2026-09-28"
    assert ambiguous is False


def test_normalize_date_ambiguous_dmy():
    cfg = Config(date_order="DMY")
    iso, ambiguous = _normalize_date("03/04/2026", cfg)
    assert iso == "2026-04-03"
    assert ambiguous is True


def test_normalize_currency_symbol():
    cfg = Config()
    assert _normalize_currency("₹", cfg) == "INR"
    assert _normalize_currency(None, cfg) == cfg.default_currency


def test_validate_catches_line_item_math_error():
    raw = {
        "vendor_name": "Acme",
        "invoice_number": "INV-1",
        "invoice_date": "2026-09-28",
        "currency": "INR",
        "line_items": [{"description": "Widget", "quantity": 10, "unit_price": 5, "amount": 100}],
        "subtotal": 100,
        "tax_total": 0,
        "discount_total": None,
        "total": 100,
    }
    result = validate(raw, ocr_text="Widget 10 5 100", cfg=Config())
    assert result.status == "needs_review"
    assert any("quantity*unit_price" in f for f in result.failed_hard_checks)


def test_validate_passes_clean_invoice():
    raw = {
        "vendor_name": "Acme",
        "invoice_number": "INV-1",
        "invoice_date": "2026-09-28",
        "currency": "INR",
        "line_items": [{"description": "Widget", "quantity": 10, "unit_price": 5, "amount": 50}],
        "subtotal": 50,
        "tax_total": 9,
        "discount_total": None,
        "total": 59,
    }
    ocr_text = "Widget 10 5 50 INV-1 59"
    result = validate(raw, ocr_text=ocr_text, cfg=Config())
    assert result.status == "ok"
    assert result.checks["line_item_math_correct"] is True
    assert result.checks["subtotal_plus_tax_equals_total"] is True
