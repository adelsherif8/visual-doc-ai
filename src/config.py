"""Central configuration: model, pricing, paths.

Read once at import. The vision model defaults to gpt-4o-mini (cheap, supports
image input, strong structured extraction) and can be overridden with the
VLM_MODEL env var.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"

VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4o-mini")

# OpenAI pricing (USD per 1M tokens) — kept local so cost is reproducible
# offline; Langfuse also computes cost server-side from usage.
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING.get(model)
    if not p:
        return 0.0
    return (prompt_tokens / 1_000_000) * p["input"] + (
        completion_tokens / 1_000_000
    ) * p["output"]
