"""Orchestration: image in -> baseline result, VLM result, annotated pages,
and a field-for-field comparison.

This is where the two engines (src/baseline.py and src/llm.py) are run against
the same page so the UI can show the gap that justifies a VLM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from PIL import Image

from . import annotate, baseline, llm, ocr
from .config import SAMPLES_DIR
from .schema import Extraction, Field


# --------------------------------------------------------------------------
# Bundled samples
# --------------------------------------------------------------------------
@dataclass
class Sample:
    name: str
    label: str
    doc_type: str
    path: str


_SAMPLE_LABELS = {
    "invoice_techflow": "🧾 SaaS Invoice",
    "receipt_greenleaf": "🛒 Store Receipt",
    "id_card_ca": "🪪 ID Card",
}


def list_samples() -> list[Sample]:
    out: list[Sample] = []
    for name, label in _SAMPLE_LABELS.items():
        meta_path = SAMPLES_DIR / f"{name}.json"
        png_path = SAMPLES_DIR / f"{name}.png"
        if meta_path.exists() and png_path.exists():
            meta = json.loads(meta_path.read_text())
            out.append(Sample(name, label, meta["doc_type"], str(png_path)))
    return out


def load_sample(name: str) -> tuple[Image.Image, dict]:
    img = Image.open(SAMPLES_DIR / f"{name}.png")
    meta = json.loads((SAMPLES_DIR / f"{name}.json").read_text())
    return img, meta


# --------------------------------------------------------------------------
# Baseline simulation (only used when tesseract is absent, so the case-study
# gap still renders on a minimal box). With OCR present we run the real thing.
# --------------------------------------------------------------------------
def _simulate_baseline(gt: dict) -> Extraction:
    keys = set(gt.get("baseline_keys", []))
    by_key = {f["key"]: f for f in gt.get("fields", [])}
    fields = [
        Field(key=k, value=by_key[k]["value"], confidence=0.68, bbox=by_key[k].get("bbox"))
        for k in keys
        if k in by_key
    ]
    return Extraction(
        engine="baseline",
        doc_type=gt.get("doc_type", "unknown"),
        fields=fields,
        note="Simulated OCR+regex baseline (tesseract not installed).",
    )


# --------------------------------------------------------------------------
# Run both engines
# --------------------------------------------------------------------------
@dataclass
class Result:
    doc_type: str
    baseline: Extraction
    vlm: Extraction
    baseline_img: Image.Image
    vlm_img: Image.Image
    comparison: list[dict]


def run(img: Image.Image, doc_type: str | None = None, ground_truth: dict | None = None) -> Result:
    ws = ocr.words(img) if ocr.available() else []
    dt = doc_type or (ground_truth or {}).get("doc_type")

    # --- baseline ---
    if ocr.available():
        base = baseline.extract(img, dt)
    elif ground_truth:
        base = _simulate_baseline(ground_truth)
    else:
        base = Extraction(engine="baseline", doc_type=dt or "unknown",
                          note="tesseract not installed — baseline unavailable.")

    # --- vlm ---
    vlm = llm.extract_vlm(img, dt or base.doc_type, ground_truth=ground_truth, ocr_words=ws)
    final_dt = vlm.doc_type if vlm.fields else base.doc_type

    return Result(
        doc_type=final_dt,
        baseline=base,
        vlm=vlm,
        baseline_img=annotate.draw(img, base.fields),
        vlm_img=annotate.draw(img, vlm.fields),
        comparison=_compare(base, vlm),
    )


def _compare(base: Extraction, vlm: Extraction) -> list[dict]:
    """One row per field the VLM found, noting whether the baseline matched."""
    b = base.by_key()
    rows: list[dict] = []
    for f in vlm.fields:
        bf = b.get(f.key)
        if bf is None:
            status = "missed"
        elif _norm(bf.value) == _norm(f.value):
            status = "match"
        else:
            status = "differs"
        rows.append({
            "field": f.label,
            "vlm": f.value,
            "baseline": bf.value if bf else "—",
            "confidence": f.confidence,
            "status": status,
        })
    return rows


def _norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())
