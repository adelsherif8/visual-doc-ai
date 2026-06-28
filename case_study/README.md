# Case study: OCR + regex vs a vision-language model

**Question:** for real-world document capture, what does a vision-language
model (VLM) actually buy you over the classic OCR-and-regex pipeline?

**Setup:** the same three documents the app ships with, run through both
engines (`src/baseline.py` and `src/llm.py`) on the same page, scored
field-for-field by `src/extract.py`. The MOCK numbers below are deterministic
and reproducible offline (`python -m src.samples` regenerates the inputs).

## Results

| Document            | Fields | Baseline (OCR+regex) | VLM (gpt-4o-mini) | Fields only the VLM got |
|---------------------|:------:|:--------------------:|:-----------------:|-------------------------|
| SaaS invoice        |   8    |          4           |         8         | Vendor, Bill To, Subtotal, Tax |
| Store receipt       |   6    |          2           |         6         | Merchant, Subtotal, Tax, Payment |
| CA driver license   |   7    |          4           |         7         | Document Type, Full Name, Address |
| **Total**           | **21** |        **10**        |      **21**       | **11 semantic fields recovered** |

Baseline field recall: **~48%**. The eleven fields it misses are not noise —
they are the ones that require *reading the layout*: telling the vendor from
the bill-to, knowing which big line is the merchant name, pulling a person's
name off an ID. Regex has no notion of any of that.

## Why the baseline misses what it misses

- **No semantics.** `INV-2026-0042` matches a pattern; "TechFlow Solutions Inc.
  is the vendor, Acme Corporation is the customer" requires understanding the
  document. Regex picks the *shape* of a value, never its *role*.
- **Layout-blind.** Two addresses on an invoice look identical to a regex.
  The VLM uses position and labels to assign each to the right field.
- **Brittle heuristics.** "The largest dollar amount is the total" works until
  a line item costs more than the total, or there are two totals.

## Where the baseline still wins

Cost and latency. The baseline is free and instant. A sensible production
pipeline often does **both**: regex/rules for the easy pattern fields, the VLM
for the semantic ones — and uses the per-field **confidence** (shown in the UI)
to decide what to auto-accept versus route to a human for review.

## Cost

gpt-4o-mini is priced at \$0.15 / \$0.60 per 1M input/output tokens. A single
page extraction is a few thousand tokens — fractions of a cent — which the app
reports live (and Langfuse records server-side when configured).
