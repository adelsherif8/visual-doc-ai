"""The baseline extractor: OCR text + regular expressions.

This is the "before" in the case study — the way document extraction was done
before vision-language models. It runs offline, costs nothing, and is genuinely
useful for clean, pattern-shaped fields (invoice numbers, dates, the total).

It is also genuinely brittle: it has no idea which line is the *vendor* vs the
*bill-to*, can't read an ID's name, and falls apart on layout it hasn't seen.
The VLM (src/llm.py) is what closes that gap — and the UI shows the difference.
"""
from __future__ import annotations

import re

from PIL import Image

from . import ocr
from .schema import Extraction, Field

_MONEY = re.compile(r"[$€£]\s?\d[\d,]*\.?\d{0,2}")
_DATE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")
_INV = re.compile(r"\b(INV[A-Z]*[-#\s:]*\d[\d-]*)", re.IGNORECASE)
_IDNUM = re.compile(r"\b[A-Z]\d{6,}\b")


def guess_doc_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("driver license", "passport", "identification", "date of birth", "dob")):
        return "id_card"
    if any(k in t for k in ("invoice", "bill to", "due date", "amount due")):
        return "invoice"
    if any(k in t for k in ("receipt", "total", "subtotal", "thank you", "change")):
        return "receipt"
    return "invoice"


def _f(key: str, value: str, ws, conf: float) -> Field:
    return Field(key=key, value=value, confidence=conf, bbox=ocr.locate(value, ws))


def extract(img: Image.Image, doc_type: str | None = None) -> Extraction:
    ws = ocr.words(img)
    text = ocr.full_text(ws)
    dt = doc_type or guess_doc_type(text)
    fields: list[Field] = []

    monies = _MONEY.findall(text)
    dates = _DATE.findall(text)

    if dt in ("invoice", "receipt"):
        if dates:
            fields.append(_f("invoice_date" if dt == "invoice" else "date", dates[0], ws, 0.71))
        if dt == "invoice":
            m = _INV.search(text)
            if m:
                fields.append(_f("invoice_number", m.group(1).strip(), ws, 0.74))
            if len(dates) > 1:
                fields.append(_f("due_date", dates[1], ws, 0.62))
        if monies:
            # naive heuristic: the largest money value is "the total"
            largest = max(monies, key=lambda s: float(re.sub(r"[^\d.]", "", s) or 0))
            fields.append(_f("total", largest, ws, 0.66))
    elif dt == "id_card":
        idm = _IDNUM.search(text)
        if idm:
            fields.append(_f("id_number", idm.group(0), ws, 0.69))
        for i, d in enumerate(dates[:3]):
            key = ("date_of_birth", "issue_date", "expiry_date")[i]
            fields.append(_f(key, d, ws, 0.6))

    note = "OCR + regex. Catches pattern-shaped fields; misses semantic ones (vendor, names, line items)."
    if not ocr.available():
        note = "tesseract not installed — baseline produced no fields. (Install tesseract-ocr to run it.)"
    return Extraction(engine="baseline", doc_type=dt, fields=fields, note=note)
