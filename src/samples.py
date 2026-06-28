"""Generate the bundled sample documents *and* their pixel-exact ground truth.

We render each document with PIL, so we know the precise bounding box of every
value as we draw it. The PNG + a `<name>.json` (doc_type, fields with bbox +
confidence, line items, and which keys a naive regex baseline would catch) are
written to data/samples/ and committed. At runtime the app just loads them —
fonts/PIL are not needed to *serve* the demo, only to (re)generate it.

    python -m src.samples          # (re)build data/samples/

This is what lets MOCK mode draw flawless boxes with zero keys and zero network.
"""
from __future__ import annotations

import hashlib
import json

from PIL import Image, ImageDraw, ImageFont

from .config import SAMPLES_DIR

INK = (17, 24, 39)
MUTED = (107, 114, 128)
LINE = (229, 231, 235)

_FONT_CANDIDATES = {
    "regular": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ],
    "bold": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ],
    "mono": [
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Courier.ttc",
    ],
}


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES[kind]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)  # PIL >=10 supports sized default


def _conf(key: str) -> float:
    """Stable, realistic-looking VLM confidence per field key (0.90–0.99)."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return round(0.90 + (h % 100) / 1000.0, 3)


class _Doc:
    """Small drawing helper that records the bbox of every value we place."""

    def __init__(self, w: int, h: int, bg=(255, 255, 255)):
        self.img = Image.new("RGB", (w, h), bg)
        self.d = ImageDraw.Draw(self.img)
        self.fields: list[dict] = []
        self.line_items: list[dict] = []
        self.words: list[dict] = []  # full text layout (for OCR-free baseline)

    def text(self, xy, s, font, fill=INK):
        self.d.text(xy, s, font=font, fill=fill)
        bbox = [int(v) for v in self.d.textbbox(xy, s, font=font)]
        self.words.append({"text": s, "bbox": bbox})
        return bbox

    def field(self, key, xy, value, font, fill=INK):
        bbox = self.text(xy, value, font, fill)
        self.fields.append({"key": key, "value": value, "bbox": bbox, "confidence": _conf(key)})
        return bbox

    def line(self, xy0, xy1, fill=LINE, width=1):
        self.d.line([xy0, xy1], fill=fill, width=width)

    def rect(self, box, fill=None, outline=None, width=1):
        self.d.rectangle(box, fill=fill, outline=outline, width=width)

    def save(self, name, doc_type, baseline_keys):
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        self.img.save(SAMPLES_DIR / f"{name}.png")
        full_text = "\n".join(w["text"] for w in self.words)
        meta = {
            "doc_type": doc_type,
            "width": self.img.width,
            "height": self.img.height,
            "fields": self.fields,
            "line_items": self.line_items,
            "baseline_keys": baseline_keys,
            "full_text": full_text,
        }
        (SAMPLES_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------
# Invoice
# --------------------------------------------------------------------------
def build_invoice():
    doc = _Doc(820, 1060)
    f_h1 = _font("bold", 30)
    f_h2 = _font("bold", 22)
    f_b = _font("regular", 16)
    f_bb = _font("bold", 16)
    f_s = _font("regular", 13)

    doc.field("vendor_name", (54, 54), "TechFlow Solutions Inc.", f_h1)
    doc.text((54, 96), "880 Innovation Way, Suite 200", f_s, MUTED)
    doc.text((54, 116), "Austin, TX 78701  ·  billing@techflow.io", f_s, MUTED)

    doc.text((600, 54), "INVOICE", f_h1, (79, 70, 229))
    doc.text((600, 100), "Invoice #", f_s, MUTED)
    doc.field("invoice_number", (600, 118), "INV-2026-0042", f_bb)
    doc.text((600, 150), "Invoice Date", f_s, MUTED)
    doc.field("invoice_date", (600, 168), "2026-06-15", f_b)
    doc.text((600, 198), "Due Date", f_s, MUTED)
    doc.field("due_date", (600, 216), "2026-07-15", f_b)

    doc.line((54, 270), (766, 270), width=2)
    doc.text((54, 290), "BILL TO", f_s, MUTED)
    doc.field("bill_to", (54, 312), "Acme Corporation", f_bb)
    doc.text((54, 336), "123 Market Street", f_s, MUTED)
    doc.text((54, 356), "San Francisco, CA 94103", f_s, MUTED)

    # line items table
    ty = 420
    doc.rect((54, ty, 766, ty + 32), fill=(243, 244, 246))
    doc.text((66, ty + 8), "Description", f_bb)
    doc.text((520, ty + 8), "Qty", f_bb)
    doc.text((660, ty + 8), "Amount", f_bb)
    rows = [
        ("Platform subscription (annual)", "1", "$4,800.00"),
        ("Onboarding & implementation", "1", "$1,500.00"),
        ("Additional API seats", "10", "$900.00"),
    ]
    y = ty + 44
    for desc, qty, amt in rows:
        db = [int(v) for v in doc.d.textbbox((66, y), desc, font=f_b)]
        doc.text((66, y), desc, f_b)
        doc.text((520, y), qty, f_b)
        ab = [int(v) for v in doc.d.textbbox((660, y), amt, font=f_b)]
        doc.text((660, y), amt, f_b)
        doc.line((54, y + 28), (766, y + 28))
        doc.line_items.append({"description": desc, "qty": qty, "amount": amt, "bbox": db + ab[2:]})
        y += 40

    # totals
    y += 12
    doc.text((560, y), "Subtotal", f_b, MUTED)
    doc.field("subtotal", (672, y), "$7,200.00", f_b)
    y += 30
    doc.text((560, y), "Tax (8.25%)", f_b, MUTED)
    doc.field("tax", (672, y), "$594.00", f_b)
    y += 34
    doc.line((540, y - 6), (766, y - 6), fill=INK)
    doc.text((560, y), "Total Due", f_bb)
    doc.field("total", (672, y), "$7,794.00", f_h2, (79, 70, 229))

    doc.text((54, 1000), "Payment due within 30 days. Thank you for your business.", f_s, MUTED)
    # regex baseline reliably catches numbers/dates/amounts, not semantic fields
    doc.save("invoice_techflow", "invoice", ["invoice_number", "invoice_date", "due_date", "total"])


# --------------------------------------------------------------------------
# Receipt
# --------------------------------------------------------------------------
def build_receipt():
    doc = _Doc(460, 760)
    f_title = _font("bold", 22)
    f_m = _font("mono", 15)
    f_mb = _font("mono", 15)  # mono "bold" approximated by same face
    f_s = _font("mono", 12)

    cx = 230

    def center(y, s, font, fill=INK):
        w = doc.d.textlength(s, font=font)
        return doc.text((cx - w / 2, y), s, font, fill)

    b = center(28, "GREENLEAF MARKET", f_title)
    doc.fields.append({"key": "merchant", "value": "GREENLEAF MARKET", "bbox": b, "confidence": _conf("merchant")})
    center(60, "412 Cedar Avenue, Portland OR", f_s, MUTED)
    center(78, "(503) 555-0147", f_s, MUTED)
    doc.line((30, 104), (430, 104))

    doc.text((30, 120), "Date:", f_m, MUTED)
    doc.field("date", (140, 120), "2026-06-12 14:32", f_m)

    doc.line((30, 150), (430, 150))
    items = [
        ("Organic Bananas 1.2kg", "$2.38"),
        ("Almond Milk 1L", "$3.49"),
        ("Sourdough Loaf", "$5.00"),
        ("Free-Range Eggs x12", "$6.29"),
        ("Dark Roast Coffee 340g", "$11.99"),
    ]
    y = 166
    for name, price in items:
        nb = [int(v) for v in doc.d.textbbox((30, y), name, font=f_m)]
        doc.text((30, y), name, f_m)
        pw = doc.d.textlength(price, font=f_m)
        pb = [int(v) for v in doc.d.textbbox((400 - pw, y), price, font=f_m)]
        doc.text((400 - pw, y), price, f_m)
        doc.line_items.append({"description": name, "qty": "1", "amount": price, "bbox": nb[:2] + pb[2:]})
        y += 26

    y += 6
    doc.line((30, y), (430, y))
    y += 14

    def row(label, key, value, fill=INK, font=f_m):
        doc.text((30, y), label, font, MUTED)
        vw = doc.d.textlength(value, font=font)
        doc.field(key, (400 - vw, y), value, font, fill)

    row("Subtotal", "subtotal", "$29.15")
    y += 26
    row("Tax (0%)", "tax", "$0.00")
    y += 30
    doc.line((30, y - 6), (430, y - 6), fill=INK)
    row("TOTAL", "total", "$29.15", (16, 122, 87), f_title)
    y += 40
    doc.text((30, y), "Paid:", f_m, MUTED)
    doc.field("payment_method", (140, y), "VISA ****1234", f_m)
    y += 40
    center(y, "*** THANK YOU ***", f_s, MUTED)
    doc.save("receipt_greenleaf", "receipt", ["date", "subtotal", "tax", "total"])


# --------------------------------------------------------------------------
# ID card
# --------------------------------------------------------------------------
def build_id_card():
    doc = _Doc(680, 430, bg=(245, 247, 250))
    f_head = _font("bold", 22)
    f_lbl = _font("regular", 11)
    f_val = _font("bold", 18)
    f_val_s = _font("bold", 15)

    doc.rect((0, 0, 680, 64), fill=(30, 58, 138))
    doc.text((24, 18), "CALIFORNIA", _font("bold", 24), (255, 255, 255))
    b = doc.text((300, 24), "DRIVER LICENSE", _font("bold", 18), (191, 219, 254))
    doc.fields.append({"key": "document_type", "value": "DRIVER LICENSE", "bbox": b, "confidence": _conf("document_type")})

    # photo placeholder
    doc.rect((24, 92, 184, 320), fill=(209, 213, 219), outline=(156, 163, 175), width=2)
    doc.text((60, 195), "PHOTO", _font("bold", 16), (107, 114, 128))

    x = 216

    def kv(y, label, key, value, font=f_val):
        doc.text((x, y), label, f_lbl, MUTED)
        doc.field(key, (x, y + 14), value, font)

    kv(96, "LN / FN", "full_name", "RIVERA, JORDAN A.")
    kv(146, "DL", "id_number", "D1234567")
    kv(196, "DOB", "date_of_birth", "1994-03-22", f_val_s)
    doc.text((430, 196), "ISS", f_lbl, MUTED)
    doc.field("issue_date", (430, 210), "2022-04-01", f_val_s)
    doc.text((560, 196), "EXP", f_lbl, MUTED)
    doc.field("expiry_date", (560, 210), "2030-03-22", f_val_s)
    doc.text((x, 250), "ADDRESS", f_lbl, MUTED)
    doc.field("address", (x, 264), "1490 Sunset Blvd, Los Angeles CA 90026", _font("regular", 14))

    doc.text((24, 360), "Class C  ·  Sex M  ·  Hgt 5-10  ·  Eyes BRN", _font("regular", 12), MUTED)
    doc.save("id_card_ca", "id_card", ["id_number", "date_of_birth", "issue_date", "expiry_date"])


def build_all():
    build_invoice()
    build_receipt()
    build_id_card()
    print(f"Wrote samples to {SAMPLES_DIR}")


if __name__ == "__main__":
    build_all()
