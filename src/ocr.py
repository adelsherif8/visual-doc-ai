"""OCR / layout layer: turn an image into word-level boxes, and locate a
given value string back to a pixel box.

Bounding boxes come from here (where text *is* on the page); the VLM decides
*what the text means*. We use pytesseract when the tesseract binary is present
and degrade gracefully to an empty layout when it isn't — MOCK mode over the
bundled samples never needs OCR (it uses baked ground truth), so the demo
always runs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image


@dataclass
class Word:
    text: str
    bbox: list[int]  # [x0, y0, x1, y1]


_TESS = None


def available() -> bool:
    """True if pytesseract + the tesseract binary are usable."""
    global _TESS
    if _TESS is None:
        try:
            import pytesseract  # noqa: F401

            pytesseract.get_tesseract_version()
            _TESS = True
        except Exception:
            _TESS = False
    return _TESS


def words(img: Image.Image) -> list[Word]:
    """Word-level boxes for an image. Empty list if OCR is unavailable."""
    if not available():
        return []
    import pytesseract

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    out: list[Word] = []
    for i, txt in enumerate(data["text"]):
        t = (txt or "").strip()
        if not t:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append(Word(t, [x, y, x + w, y + h]))
    return out


def full_text(ws: list[Word]) -> str:
    return " ".join(w.text for w in ws)


_NORM = re.compile(r"[^a-z0-9]")


def _norm(s: str) -> str:
    return _NORM.sub("", s.lower())


def locate(value: str, ws: list[Word]) -> list[int] | None:
    """Find the pixel box covering `value` by matching it to a run of words.

    Tries the longest contiguous run of OCR words whose concatenation contains
    the normalized value (handles values split across tokens like "$7,794.00").
    Returns the union box, or None if nothing reasonable matches.
    """
    if not ws or not value.strip():
        return None
    target = _norm(value)
    if not target:
        return None

    n = len(ws)
    best: tuple[int, list[int]] | None = None  # (run_len, box)
    for i in range(n):
        acc = ""
        for j in range(i, min(i + 8, n)):
            acc += _norm(ws[j].text)
            if target in acc or acc in target and len(acc) >= max(3, len(target) // 2):
                box = _union(ws[i : j + 1])
                run = j - i + 1
                # prefer the match whose text is closest in length to target
                score = -abs(len(acc) - len(target))
                if best is None or score > best[0]:
                    best = (score, box)
                if target in acc:
                    break
    return best[1] if best else None


def _union(ws: list[Word]) -> list[int]:
    xs0 = min(w.bbox[0] for w in ws)
    ys0 = min(w.bbox[1] for w in ws)
    xs1 = max(w.bbox[2] for w in ws)
    ys1 = max(w.bbox[3] for w in ws)
    return [xs0, ys0, xs1, ys1]
