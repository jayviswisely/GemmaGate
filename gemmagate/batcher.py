"""Sentiment batcher — amortize instruction overhead across residues.

Sentiment tasks the lexicon abstained on each cost a full instruction header
(~25 tokens) plus their text if sent individually. Batching N of them into
one numbered call pays the header once and caps output at ~4 tokens/item.

Accuracy protections:
  * only label-only tasks are batched (justifications stay individual)
  * tasks are grouped by IDENTICAL allowed-label sets
  * every returned line is validated per-task; unparsed/invalid items are
    returned to the caller for the normal individual ladder — a batch can
    reduce tokens but can never reduce accuracy below the individual path
  * chunk size capped (6) to keep line-mapping errors near zero
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from .prompts import build_sentiment_batch
from .remote import FireworksClient
from .schemas import Attempt, Route, Solved, TaskSpec, Validation
from .validator import validate

log = logging.getLogger("gemmagate.batcher")

_LINE_RE = re.compile(r"^\s*(\d+)\s*[:.\)-]\s*(.+?)\s*$")
_CHUNK = 6
_DEFAULT_LABELS = ("positive", "negative", "neutral", "mixed")


class SentimentBatcher:
    def __init__(self, client: FireworksClient, cheap_model: Optional[str]):
        self.client = client
        self.cheap_model = cheap_model

    def eligible(self, spec: TaskSpec) -> bool:
        return (not spec.wants_justification and not spec.wants_json
                and bool(spec.payload or spec.prompt))

    def solve(self, items: list[tuple[int, TaskSpec]],
              deadline: float) -> dict[int, Solved]:
        """items: (input_index, spec). Returns index -> Solved for the subset
        confidently answered; everything else falls back to the caller."""
        if not self.cheap_model or len(items) < 2:
            return {}
        out: dict[int, Solved] = {}

        # group by identical label sets so one header fits all items
        groups: dict[tuple, list[tuple[int, TaskSpec]]] = {}
        for idx, spec in items:
            key = tuple(l.lower() for l in (spec.allowed_labels or _DEFAULT_LABELS))
            groups.setdefault(key, []).append((idx, spec))

        for labels, group in groups.items():
            for chunk_start in range(0, len(group), _CHUNK):
                chunk = group[chunk_start:chunk_start + _CHUNK]
                if len(chunk) < 2 or time.time() > deadline - 15:
                    continue
                out.update(self._solve_chunk(chunk, list(labels)))
        return out

    def _solve_chunk(self, chunk: list[tuple[int, TaskSpec]],
                     labels: list[str]) -> dict[int, Solved]:
        numbered = [(n + 1, spec.payload or spec.prompt)
                    for n, (_, spec) in enumerate(chunk)]
        prompt, max_tok = build_sentiment_batch(numbered, labels)
        t0 = time.time()
        res = self.client.complete(self.cheap_model, prompt, max_tokens=max_tok)
        if not res.text:
            return {}

        by_number: dict[int, str] = {}
        for line in res.text.splitlines():
            m = _LINE_RE.match(line)
            if m:
                by_number[int(m.group(1))] = m.group(2)

        per_item_tokens = (res.input_tokens + res.output_tokens) // max(len(chunk), 1)
        out: dict[int, Solved] = {}
        for n, (idx, spec) in enumerate(chunk, start=1):
            raw = by_number.get(n)
            if raw is None:
                continue                      # unparsed -> individual fallback
            v = validate(spec, raw)
            if not v.passed:
                continue                      # invalid -> individual fallback
            attempt = Attempt(Route.REMOTE_CHEAP, res.model,
                              v.repaired or raw, v,
                              input_tokens=per_item_tokens, output_tokens=0)
            out[idx] = Solved(task_id=spec.task_id,
                              answer=attempt.answer, route=Route.REMOTE_CHEAP,
                              category=spec.category, risk=spec.risk,
                              confidence=v.score, attempts=[attempt],
                              remote_tokens=per_item_tokens,
                              wall_time_s=time.time() - t0)
        log.info("batch of %d: %d accepted, %d fallback",
                 len(chunk), len(out), len(chunk) - len(out))
        return out
