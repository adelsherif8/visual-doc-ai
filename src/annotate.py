"""Draw extracted fields back onto the page as confidence-colored boxes.

Green = high confidence, amber = medium, red = low. Each box gets a small label
tag with the field name. This is the visual payoff of the demo: you *see* what
the model found and where.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .schema import Field

_GREEN = (16, 185, 129)
_AMBER = (217, 119, 6)
_RED = (220, 38, 38)


def _color(conf: float) -> tuple[int, int, int]:
    if conf >= 0.85:
        return _GREEN
    if conf >= 0.65:
        return _AMBER
    return _RED


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def draw(img: Image.Image, fields: list[Field]) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out, "RGBA")
    font = _font(13)
    for f in fields:
        if not f.bbox:
            continue
        x0, y0, x1, y1 = f.bbox
        c = _color(f.confidence)
        # translucent fill + solid border
        d.rectangle([x0 - 3, y0 - 2, x1 + 3, y1 + 2], fill=c + (38,), outline=c, width=2)
        # label tag above (or below if no room)
        tag = f.label
        tw = d.textlength(tag, font=font)
        ty = y0 - 19 if y0 - 19 > 0 else y1 + 3
        d.rectangle([x0 - 3, ty, x0 - 3 + tw + 10, ty + 17], fill=c)
        d.text((x0 + 2, ty + 2), tag, fill=(255, 255, 255), font=font)
    return out
