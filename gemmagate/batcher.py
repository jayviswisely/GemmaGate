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


class ShortAnswerBatcher:
    """One numbered remote call for several short factual questions.

    Each escalated factual task normally pays its own instruction overhead
    (~30-45 input tokens). Batching shares that overhead across the group:
    N questions -> one call. Any item that fails per-item validation simply
    falls through to the normal individual ladder, so batching can only
    save tokens, never cost accuracy.
    """

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def solve(self, candidates, deadline):
        import time as _t
        from .schemas import Route, Solved
        from .validator import validate
        if not self.model or _t.time() > deadline - 10 or getattr(self.client, "dry_run", False):
            return {}
        numbered = "\n".join(f"{k+1}. {s.prompt.strip()}"
                              for k, (_, s) in enumerate(candidates))
        prompt = ("Answer each numbered question in one concise, complete "
                  "sentence. Output ONLY the numbered answers, one per line.\n"
                  + numbered)
        max_tok = 20 + 40 * len(candidates)
        t0 = time.time()
        try:
            res = self.client.complete(self.model, prompt, max_tokens=max_tok)
        except Exception:
            return {}
        if not res or not res.text:
            return {}
        answers = {}
        for line in res.text.splitlines():
            m = re.match(r"\s*(\d+)[.):]\s*(.+)", line)
            if m:
                answers[int(m.group(1))] = m.group(2).strip()
        out = {}
        total = getattr(res, "total_tokens",
                        getattr(res, "input_tokens", 0)
                        + getattr(res, "output_tokens", 0))
        per_task = max(total, 0) // max(len(candidates), 1)
        for k, (i, s) in enumerate(candidates):
            a = answers.get(k + 1, "")
            v = validate(s, a)
            if a and v.passed:
                out[i] = Solved(task_id=s.task_id, answer=v.repaired or a,
                                category=s.category, route=Route.REMOTE_CHEAP,
                                confidence=v.score, remote_tokens=per_task,
                                wall_time_s=time.time() - t0)
        return out
