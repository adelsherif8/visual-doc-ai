"""The VLM extraction layer — the "after" in the case study.

A single `extract_vlm(...)` returns an `Extraction` whose fields carry values,
per-field confidence, and (via the OCR layer) pixel boxes.

  * Real mode  -> gpt-4o-mini with the page image + a strict JSON instruction.
                  Langfuse drop-in wrapper when LANGFUSE_* keys are present, so
                  every extraction is auto-traced with token + cost usage.
  * Mock mode  -> for the bundled samples, returns the baked ground truth (so
                  boxes + JSON are pixel-perfect with NO key, network, or cost).
                  For an arbitrary upload with no key it returns nothing and
                  asks for a key — honest about what runs offline.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os

from PIL import Image

from . import ocr
from .config import VLM_MODEL, cost_usd
from .schema import DOC_FIELDS, Extraction, Field, LineItem

MOCK = not bool(os.getenv("OPENAI_API_KEY"))
_LANGFUSE_ON = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
LANGFUSE_ENABLED = _LANGFUSE_ON

if not _LANGFUSE_ON:
    logging.getLogger("langfuse").setLevel(logging.CRITICAL)

_client = None
if not MOCK:
    if _LANGFUSE_ON:
        from langfuse.openai import OpenAI  # type: ignore
    else:
        from openai import OpenAI  # type: ignore
    _client = OpenAI()

_MOCK_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def status() -> dict:
    return {
        "mock": MOCK,
        "langfuse": _LANGFUSE_ON,
        "model": VLM_MODEL,
        "ocr": ocr.available(),
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def extract_vlm(
    img: Image.Image,
    doc_type: str | None,
    ground_truth: dict | None = None,
    ocr_words: list | None = None,
) -> Extraction:
    if MOCK:
        return _mock(doc_type, ground_truth)
    return _live(img, doc_type, ocr_words)


# --------------------------------------------------------------------------
# MOCK
# --------------------------------------------------------------------------
def _mock(doc_type: str | None, ground_truth: dict | None) -> Extraction:
    if not ground_truth:
        return Extraction(
            engine="vlm",
            doc_type=doc_type or "unknown",
            fields=[],
            usage=dict(_MOCK_USAGE),
            note="MOCK mode: add OPENAI_API_KEY to run the VLM on uploaded documents.",
        )
    fields = [
        Field(key=f["key"], value=f["value"], confidence=f.get("confidence", 0.95), bbox=f.get("bbox"))
        for f in ground_truth.get("fields", [])
    ]
    items = [
        LineItem(li["description"], li.get("qty", ""), li.get("amount", ""), li.get("bbox"))
        for li in ground_truth.get("line_items", [])
    ]
    return Extraction(
        engine="vlm",
        doc_type=ground_truth.get("doc_type", doc_type or "unknown"),
        fields=fields,
        line_items=items,
        usage=dict(_MOCK_USAGE),
        note="MOCK (deterministic): bundled sample with baked ground truth.",
    )


# --------------------------------------------------------------------------
# LIVE
# --------------------------------------------------------------------------
def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _prompt(doc_type: str | None) -> str:
    known = ""
    if doc_type and doc_type in DOC_FIELDS:
        keys = ", ".join(k for k, _ in DOC_FIELDS[doc_type])
        known = f"This looks like a {doc_type}. Prefer these field keys when present: {keys}.\n"
    return (
        "You are a precise document data-extraction engine. Read the document image and "
        "return STRICT JSON only, no prose.\n"
        f"{known}"
        "Shape:\n"
        '{ "doc_type": "invoice|receipt|id_card|other",\n'
        '  "fields": [ { "key": "snake_case_field", "value": "string", "confidence": 0.0-1.0 } ],\n'
        '  "line_items": [ { "description": "", "qty": "", "amount": "" } ] }\n'
        "Use snake_case keys. Confidence reflects how sure you are of each value. "
        "Copy values verbatim as they appear. Omit fields that are not present. "
        "line_items only for invoices/receipts."
    )


def _live(img: Image.Image, doc_type: str | None, ocr_words: list | None) -> Extraction:
    resp = _client.chat.completions.create(
        model=VLM_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _prompt(doc_type)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract this document."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(img)}"}},
                ],
            },
        ],
    )
    msg = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(msg)
    except json.JSONDecodeError:
        parsed = {"doc_type": doc_type or "unknown", "fields": []}

    ws = ocr_words if ocr_words is not None else ocr.words(img)
    fields: list[Field] = []
    for f in parsed.get("fields", []):
        key, value = f.get("key"), f.get("value")
        if not key or value in (None, ""):
            continue
        fields.append(
            Field(
                key=key,
                value=str(value),
                confidence=float(f.get("confidence", 0.8)),
                bbox=ocr.locate(str(value), ws),
            )
        )
    items = [
        LineItem(
            str(li.get("description", "")),
            str(li.get("qty", "")),
            str(li.get("amount", "")),
            ocr.locate(str(li.get("description", "")), ws),
        )
        for li in parsed.get("line_items", [])
        if li.get("description")
    ]

    u = resp.usage
    usage = {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "cost_usd": cost_usd(VLM_MODEL, u.prompt_tokens, u.completion_tokens),
    }
    return Extraction(
        engine="vlm",
        doc_type=parsed.get("doc_type", doc_type or "unknown"),
        fields=fields,
        line_items=items,
        usage=usage,
        note=f"Live extraction via {VLM_MODEL}.",
    )
