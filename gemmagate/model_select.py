"""Model selection — rank ALLOWED_MODELS into cheap/mid/strong tiers at
runtime using only name hints. No model IDs are hardcoded anywhere.

Heuristics (conservative, config-overridable):
  * parameter count in the name ("8b", "70b", "1.5B", "8x7b") => size score
  * keywords: nano/tiny/mini/small lower the score; large/big raise it
  * "instruct"/"chat" variants are preferred over base models at equal size
  * unknown names get a middle default so they're never accidentally "cheap"

If only one model is allowed, all tiers point at it. If two, cheap=smallest,
strong=largest, mid=strong. Overrides in config/settings.yaml (tier_overrides)
let you pin tiers by substring on launch day without code changes.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger("gemmagate.models")

_MOE_RE = re.compile(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b", re.I)
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.I)
_KEYWORDS = [
    ("nano", 0.5), ("tiny", 1.0), ("mini", 2.0), ("small", 3.0),
    ("lite", 3.0), ("medium", 20.0), ("large", 70.0), ("big", 70.0),
    ("xl", 70.0), ("max", 100.0), ("ultra", 200.0),
]
_DEFAULT_SIZE = 30.0   # unknown => assume mid-size (never assume cheap)


def estimate_size_b(name: str) -> float:
    low = name.lower()
    m = _MOE_RE.search(low)
    if m:  # mixture-of-experts: experts * size approximates capability cost
        return float(m.group(1)) * float(m.group(2))
    m = _SIZE_RE.search(low)
    if m:
        return float(m.group(1))
    for kw, size in _KEYWORDS:
        if kw in low:
            return size
    return _DEFAULT_SIZE


def _pref_bonus(name: str) -> float:
    low = name.lower()
    bonus = 0.0
    if "instruct" in low or "chat" in low or "-it" in low:
        bonus += 0.1
    if "base" in low:
        bonus -= 0.5
    return bonus


@dataclass
class TierPlan:
    cheap: str
    mid: str
    strong: str
    sizes: dict

    def as_dict(self) -> dict:
        return {"cheap": self.cheap, "mid": self.mid, "strong": self.strong}


def plan_tiers(allowed_models: list[str],
               overrides: dict | None = None) -> TierPlan:
    models = [m.strip() for m in allowed_models if m and m.strip()]
    if not models:
        raise ValueError("ALLOWED_MODELS is empty")

    overrides = overrides or {}

    def find_override(tier: str):
        sub = overrides.get(tier)
        if not sub:
            return None
        for m in models:
            if sub.lower() in m.lower():
                return m
        return None

    ranked = sorted(models, key=lambda m: (estimate_size_b(m), -_pref_bonus(m)))
    sizes = {m: estimate_size_b(m) for m in models}

    cheap = find_override("cheap") or ranked[0]
    strong = find_override("strong") or ranked[-1]
    if len(ranked) >= 3:
        mid_default = ranked[len(ranked) // 2]
        if mid_default in (cheap, strong) and len(ranked) > 3:
            mid_default = next((m for m in ranked[1:-1] if m not in (cheap, strong)),
                               strong)
        mid = find_override("mid") or mid_default
    else:
        mid = find_override("mid") or strong

    plan = TierPlan(cheap=cheap, mid=mid, strong=strong, sizes=sizes)
    log.info("model tiers: cheap=%s mid=%s strong=%s (sizes=%s)",
             cheap, mid, strong, sizes)
    return plan
