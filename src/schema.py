"""The shape of an extraction result.

A document becomes a flat list of `Field`s (key/value/confidence/bbox) plus an
optional `line_items` table. Both the deterministic baseline and the VLM emit
this same shape so the UI can diff them field-for-field.

`bbox` is [x0, y0, x1, y1] in *image pixels*. It is what we draw on the page.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The fields we try to pull per document type. The VLM is told this list so its
# output keys line up with the baseline's; the UI uses the labels for display.
DOC_FIELDS: dict[str, list[tuple[str, str]]] = {
    "invoice": [
        ("invoice_number", "Invoice #"),
        ("invoice_date", "Invoice Date"),
        ("due_date", "Due Date"),
        ("vendor_name", "Vendor"),
        ("bill_to", "Bill To"),
        ("subtotal", "Subtotal"),
        ("tax", "Tax"),
        ("total", "Total"),
    ],
    "receipt": [
        ("merchant", "Merchant"),
        ("date", "Date"),
        ("payment_method", "Payment"),
        ("subtotal", "Subtotal"),
        ("tax", "Tax"),
        ("total", "Total"),
    ],
    "id_card": [
        ("document_type", "Document Type"),
        ("full_name", "Full Name"),
        ("id_number", "ID Number"),
        ("date_of_birth", "Date of Birth"),
        ("issue_date", "Issue Date"),
        ("expiry_date", "Expiry Date"),
        ("address", "Address"),
    ],
}

LABELS: dict[str, str] = {k: lbl for fields in DOC_FIELDS.values() for k, lbl in fields}


@dataclass
class Field:
    key: str
    value: str
    confidence: float
    bbox: list[int] | None = None  # [x0, y0, x1, y1] in image px

    @property
    def label(self) -> str:
        return LABELS.get(self.key, self.key.replace("_", " ").title())

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.key,
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
        }


@dataclass
class LineItem:
    description: str
    qty: str = ""
    amount: str = ""
    bbox: list[int] | None = None


@dataclass
class Extraction:
    engine: str  # "baseline" | "vlm"
    doc_type: str
    fields: list[Field] = field(default_factory=list)
    line_items: list[LineItem] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def by_key(self) -> dict[str, Field]:
        return {f.key: f for f in self.fields}

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "doc_type": self.doc_type,
            "fields": {f.key: f.value for f in self.fields},
        }
        if self.line_items:
            out["line_items"] = [
                {"description": li.description, "qty": li.qty, "amount": li.amount}
                for li in self.line_items
            ]
        return out
