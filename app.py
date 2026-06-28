"""Gradio UI for Visual Document AI.

Pick a bundled document (or upload your own) → the page comes back with
confidence-colored boxes drawn over every detected field, clean structured
JSON, and a side-by-side of the OCR+regex baseline vs the VLM so you can see
exactly what a vision-language model buys you.

Runs with no keys in MOCK mode (deterministic, over the bundled samples).
Set OPENAI_API_KEY for live extraction on any upload; LANGFUSE_* for traces.
"""
from __future__ import annotations

import html
import json

import gradio as gr

from src import extract
from src.llm import status

ST = status()
SAMPLES = extract.list_samples()
_BY_LABEL = {s.label: s for s in SAMPLES}

ENGINE_VLM = f"VLM ({ST['model']})"
ENGINE_BASE = "Baseline (OCR + regex)"

CSS = """
.sq{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:middle;margin-right:3px}
.kpi{display:inline-block;background:#f3f4f6;border-radius:999px;padding:4px 14px;
  font-size:12px;margin:0 6px 6px 0;color:#374151;font-variant-numeric:tabular-nums}
.kpi b{color:#111827}
.kpi.win{background:#ecfdf5;color:#065f46}
.fld{border:1px solid #e5e7eb;border-radius:10px;padding:8px 12px;margin:6px 0;background:#fff}
.fld-top{display:flex;justify-content:space-between;font-size:13px}
.fld-key{color:#6b7280}
.fld-val{font-weight:600;color:#111827;font-family:ui-monospace,Menlo,monospace}
.bar{height:6px;border-radius:999px;background:#f3f4f6;margin-top:6px;overflow:hidden}
.bar > span{display:block;height:100%}
.note{color:#6b7280;font-size:12px;padding:6px 2px}
table.cmp{border-collapse:collapse;width:100%;font-size:13px}
table.cmp th,table.cmp td{border:1px solid #e5e7eb;padding:6px 9px;text-align:left;vertical-align:top}
table.cmp th{background:#f9fafb}
.tag{font-size:11px;font-weight:700;padding:1px 8px;border-radius:999px}
.tag.match{background:#ecfdf5;color:#065f46}
.tag.missed{background:#fef2f2;color:#991b1b}
.tag.differs{background:#fffbeb;color:#92400e}
.mono{font-family:ui-monospace,Menlo,monospace}
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _conf_color(c: float) -> str:
    return "#10b981" if c >= 0.85 else "#d97706" if c >= 0.65 else "#dc2626"


def _fields_html(ext) -> str:
    if not ext.fields:
        return f"<div class='note'>{_esc(ext.note)}</div>"
    rows = []
    for f in ext.fields:
        pct = int(round(f.confidence * 100))
        rows.append(
            f"<div class='fld'><div class='fld-top'>"
            f"<span class='fld-key'>{_esc(f.label)}</span>"
            f"<span class='fld-val'>{_esc(f.value)}</span></div>"
            f"<div class='bar'><span style='width:{pct}%;background:{_conf_color(f.confidence)}'></span></div>"
            f"<div class='note' style='padding:2px 0 0'>confidence {pct}%</div></div>"
        )
    return f"<div class='note'>{_esc(ext.note)}</div>" + "".join(rows)


def _kpi_html(result) -> str:
    b, v = result.baseline, result.vlm
    missed = sum(1 for c in result.comparison if c["status"] == "missed")
    cost = v.usage.get("cost_usd", 0.0)
    tok = v.usage.get("prompt_tokens", 0) + v.usage.get("completion_tokens", 0)
    return (
        f"<span class='kpi'>doc type <b>{_esc(result.doc_type)}</b></span>"
        f"<span class='kpi'>baseline fields <b>{len(b.fields)}</b></span>"
        f"<span class='kpi win'>VLM fields <b>{len(v.fields)}</b></span>"
        f"<span class='kpi win'>recovered by VLM <b>+{missed}</b></span>"
        f"<span class='kpi'>tokens <b>{tok}</b></span>"
        f"<span class='kpi'>cost <b>${cost:.5f}</b></span>"
    )


def _compare_html(result) -> str:
    if not result.comparison:
        return "<div class='note'>No VLM fields to compare. (MOCK upload — add OPENAI_API_KEY.)</div>"
    head = "<tr><th>Field</th><th>VLM value</th><th>Baseline (OCR+regex)</th><th>Conf.</th><th></th></tr>"
    body = []
    for c in result.comparison:
        tag = c["status"]
        base = c["baseline"] if tag != "missed" else "— not found —"
        body.append(
            f"<tr><td>{_esc(c['field'])}</td>"
            f"<td class='mono'>{_esc(c['vlm'])}</td>"
            f"<td class='mono'>{_esc(base)}</td>"
            f"<td>{int(round(c['confidence']*100))}%</td>"
            f"<td><span class='tag {tag}'>{tag}</span></td></tr>"
        )
    return f"<table class='cmp'>{head}{''.join(body)}</table>"


def _json_str(ext) -> str:
    return json.dumps(ext.as_dict(), indent=2)


def _pick_image(result, engine):
    if result is None:
        return None, ""
    if engine == ENGINE_BASE:
        return result.baseline_img, _fields_html(result.baseline)
    return result.vlm_img, _fields_html(result.vlm)


def run(sample_label, upload, engine):
    if upload is not None:
        result = extract.run(upload)
    elif sample_label:
        s = _BY_LABEL[sample_label]
        img, gt = extract.load_sample(s.name)
        result = extract.run(img, ground_truth=gt)
    else:
        return None, "<div class='note'>Pick a sample or upload a document, then Extract.</div>", "", "{}", "", None
    img, fields = _pick_image(result, engine)
    return img, fields, _kpi_html(result), _json_str(result.vlm), _compare_html(result), result


def switch_engine(result, engine):
    img, fields = _pick_image(result, engine)
    return img, fields


def load_preview(sample_label):
    if not sample_label:
        return None
    s = _BY_LABEL[sample_label]
    return s.path


MODE = "LIVE (OpenAI)" if not ST["mock"] else "MOCK (no key — deterministic offline demo)"
OCR = "OCR on (tesseract)" if ST["ocr"] else "OCR off (install tesseract for live uploads)"
LF = "Langfuse tracing ON" if ST["langfuse"] else "Langfuse off"

with gr.Blocks(title="Visual Document AI") as demo:
    gr.HTML(f"<style>{CSS}</style>")
    gr.Markdown(
        f"# Visual Document AI — extract structured data from any document\n"
        f"Upload an invoice, receipt, or ID → get **boxes drawn over every field**, "
        f"**structured JSON with confidence**, and a **baseline-vs-VLM** comparison.\n\n"
        f"`{MODE}` · model `{ST['model']}` · {OCR} · {LF}"
    )
    state = gr.State()
    with gr.Row():
        with gr.Column(scale=2):
            sample = gr.Radio(
                [s.label for s in SAMPLES], label="Bundled sample documents",
                value=SAMPLES[0].label if SAMPLES else None,
            )
            upload = gr.Image(label="…or upload your own (PNG/JPG)", type="pil", height=160)
            engine = gr.Radio([ENGINE_VLM, ENGINE_BASE], value=ENGINE_VLM, label="Show boxes from")
            run_btn = gr.Button("Extract fields", variant="primary")
            gr.Markdown(
                "<span class='sq' style='background:#10b981'></span> high &nbsp;"
                "<span class='sq' style='background:#d97706'></span> medium &nbsp;"
                "<span class='sq' style='background:#dc2626'></span> low confidence. "
                "Boxes are grounded in the page via the OCR layer; the VLM decides "
                "what each value *means*."
            )
        with gr.Column(scale=3):
            kpi = gr.HTML()
            out_img = gr.Image(label="Annotated document", height=560)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Detected fields")
            fields_html = gr.HTML()
        with gr.Column():
            with gr.Tab("Structured JSON"):
                json_out = gr.Code(value="{}", language="json", label="VLM output")
            with gr.Tab("Baseline vs VLM"):
                compare = gr.HTML()

    sample.change(load_preview, sample, out_img)
    run_btn.click(run, [sample, upload, engine], [out_img, fields_html, kpi, json_out, compare, state])
    engine.change(switch_engine, [state, engine], [out_img, fields_html])

if __name__ == "__main__":
    demo.launch()
