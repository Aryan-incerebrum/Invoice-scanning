"""Stage 4: normalize and validate. The validation gate, not the LLM, decides
whether a result is trusted — see design doc section 3.

One public function: validate(raw_invoice, ocr_text, cfg) -> ValidationResult.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from .config import Config

_ISO_CURRENCY_CODES = {
    "INR", "USD", "EUR", "GBP", "AED", "AUD", "CAD", "JPY", "CNY", "SGD",
}  # extend as needed — kept short deliberately rather than importing a huge table


def _to_number(raw, cfg: Config) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    # strip currency symbols/letters, keep digits, separators, sign
    s = re.sub(r"[^\d.,\-]", "", s)
    if cfg.thousands_separator == "," and cfg.decimal_separator == ".":
        s = s.replace(",", "")
    elif cfg.thousands_separator == "." and cfg.decimal_separator == ",":
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_date(raw, cfg: Config) -> tuple[Optional[str], bool]:
    """Returns (iso_date_or_None, ambiguous_flag)."""
    if not raw:
        return None, False
    s = str(raw).strip()

    iso_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if iso_match:
        return s, False

    m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$", s)
    if not m:
        return None, False  # unparseable — leave null, flag happens in validate()

    a, b, year = int(m.group(1)), int(m.group(2)), m.group(3)
    year = int(year) if len(year) == 4 else 2000 + int(year)

    ambiguous = a <= 12 and b <= 12 and a != b
    if cfg.date_order == "DMY":
        day, month = a, b
    else:
        month, day = a, b

    try:
        return date(year, month, day).isoformat(), ambiguous
    except ValueError:
        return None, False


def _normalize_currency(raw, cfg: Config) -> Optional[str]:
    if not raw:
        return cfg.default_currency
    s = str(raw).strip().upper()
    if s in _ISO_CURRENCY_CODES:
        return s
    symbol_map = {"₹": "INR", "RS": "INR", "RS.": "INR", "$": "USD", "€": "EUR", "£": "GBP"}
    return symbol_map.get(s, cfg.default_currency)


@dataclass
class ValidationResult:
    status: str                      # ok | ok_with_warnings | needs_review
    invoice: dict
    checks: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failed_hard_checks: list[str] = field(default_factory=list)


def validate(raw_invoice: dict, ocr_text: str, cfg: Config = Config()) -> ValidationResult:
    inv = dict(raw_invoice)
    warnings: list[str] = []

    inv["invoice_date"], date_ambiguous = _normalize_date(inv.get("invoice_date"), cfg)
    if date_ambiguous:
        warnings.append("invoice_date was ambiguous (day/month order); used configured locale default")
    inv["due_date"], _ = _normalize_date(inv.get("due_date"), cfg)
    inv["currency"] = _normalize_currency(inv.get("currency"), cfg)

    for money_field in ("subtotal", "tax_total", "discount_total", "total"):
        inv[money_field] = _to_number(inv.get(money_field), cfg)

    line_items = []
    for item in inv.get("line_items", []):
        item = dict(item)
        item["quantity"] = _to_number(item.get("quantity"), cfg)
        item["unit_price"] = _to_number(item.get("unit_price"), cfg)
        item["amount"] = _to_number(item.get("amount"), cfg)
        line_items.append(item)
    inv["line_items"] = line_items

    # --- hard checks ---
    failed_hard: list[str] = []
    tol = cfg.arithmetic_tolerance

    line_sums_ok = True
    for i, item in enumerate(line_items):
        q, u, a = item.get("quantity"), item.get("unit_price"), item.get("amount")
        if None in (q, u, a):
            continue  # can't check an incomplete row; not itself a hard failure
        if abs(q * u - a) > tol:
            line_sums_ok = False
            failed_hard.append(f"line_item[{i}]: quantity*unit_price ({q*u:.2f}) != amount ({a:.2f})")

    subtotal = inv.get("subtotal")
    computed_subtotal = sum(li["amount"] for li in line_items if li.get("amount") is not None)
    line_sums_match_subtotal = True
    if subtotal is not None and line_items:
        line_sums_match_subtotal = abs(computed_subtotal - subtotal) <= tol
        if not line_sums_match_subtotal:
            failed_hard.append(f"line items sum to {computed_subtotal:.2f}, subtotal says {subtotal:.2f}")

    subtotal_plus_tax_equals_total = True
    total = inv.get("total")
    tax = inv.get("tax_total") or 0.0
    discount = inv.get("discount_total") or 0.0
    if subtotal is not None and total is not None:
        subtotal_plus_tax_equals_total = abs((subtotal + tax - discount) - total) <= tol
        if not subtotal_plus_tax_equals_total:
            failed_hard.append(
                f"subtotal+tax-discount ({subtotal + tax - discount:.2f}) != total ({total:.2f})"
            )

    required_present = all(
        inv.get(f) not in (None, "") for f in ("vendor_name", "invoice_number", "invoice_date", "total")
    )
    if not required_present:
        failed_hard.append("a required field is missing")

    # --- soft checks ---
    def _number_candidates(value: float) -> list[str]:
        """Plausible printed forms of a normalized float, so '59.0' matches
        a printed '59' and '1250.0' matches a printed '1,250.00'."""
        candidates = []
        if float(value).is_integer():
            candidates.append(str(int(value)))
        candidates.append(f"{value:.2f}")
        candidates.append(f"{value:g}")
        return candidates

    def _appears_in_ocr(value) -> bool:
        if value is None:
            return True  # nothing to cross-check
        haystack_digits_free = ocr_text  # keep as-is; we match against printed punctuation too
        if isinstance(value, (int, float)):
            return any(c in ocr_text for c in _number_candidates(float(value)))
        needle = str(value).strip()
        return needle in haystack_digits_free if needle else True

    values_found = all(_appears_in_ocr(inv.get(f)) for f in ("invoice_number", "total")) and all(
        _appears_in_ocr(li.get("amount")) for li in line_items
    )
    if not values_found:
        warnings.append("one or more extracted numbers were not found verbatim in the OCR text — possible hallucination")

    today = date.today()
    if inv.get("invoice_date"):
        try:
            if datetime.fromisoformat(inv["invoice_date"]).date() > today:
                warnings.append("invoice_date is in the future")
        except ValueError:
            pass

    if cfg.known_vendors and inv.get("vendor_name") not in cfg.known_vendors:
        warnings.append(f"vendor '{inv.get('vendor_name')}' is not in the known-vendor list")

    checks = {
        "line_item_math_correct": line_sums_ok,
        "line_sums_match_subtotal": line_sums_match_subtotal,
        "subtotal_plus_tax_equals_total": subtotal_plus_tax_equals_total,
        "required_fields_present": required_present,
        "values_found_in_ocr_text": values_found,
    }

    if failed_hard:
        status = "needs_review"
    elif warnings:
        status = "ok_with_warnings"
    else:
        status = "ok"

    return ValidationResult(status=status, invoice=inv, checks=checks, warnings=warnings, failed_hard_checks=failed_hard)
