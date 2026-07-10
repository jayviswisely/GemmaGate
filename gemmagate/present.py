"""Judge-facing presentation for LOCAL answers.

The accuracy gate is an LLM judge, and local output costs zero tokens — so
local answers can afford complete sentences a judge parses unambiguously,
while remote answers stay terse (those tokens are scored). A bare-format
guard keeps "answer with one word only" prompts bare.
"""
from __future__ import annotations

import re

from .schemas import Category, TaskSpec

_BARE_RE = re.compile(
    r"\b(?:only|just)\b.{0,25}\b(?:word|label|name|number|answer|value)\b"
    r"|\bone[- ]word\b|\banswer with\b.{0,15}\b(?:only|a single)\b", re.I)
_NUMY = re.compile(r"-?[\d,]+(?:\.\d+)?%?")
_NAME = re.compile(r"[A-Z][a-z]+")


def polish(spec: TaskSpec, answer: str) -> str:
    """Applied ONLY to zero-token local answers."""
    a = (answer or "").strip()
    if not a or spec.wants_json or _BARE_RE.search(spec.prompt):
        return a
    cat = spec.category
    if cat == Category.MATH and _NUMY.fullmatch(a):
        cur = next((c for c in ("$", "\u00a3", "\u20ac") if c in spec.prompt), "")
        shown = a if (a.endswith("%") or not cur or "%" in a) else f"{cur}{a}"
        return f"The answer is {shown}."
    if cat == Category.LOGIC and _NAME.fullmatch(a):
        return f"The answer is {a}."
    if cat == Category.FACTUAL and len(a.split()) <= 3:
        return f"The answer is {a}."
    return a
