"""Router — orchestrates the full run within the 10-minute budget.

  * builds tiers from ALLOWED_MODELS at startup
  * classifies + risk-scores every task (free) before any remote call
  * solves concurrently (remote-bound tasks overlap network latency)
  * per-task exception isolation: one bad task never poisons the batch
  * deadline manager: tasks that would start too late get failsafe answers
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .batcher import SentimentBatcher, ShortAnswerBatcher
from .present import polish
from .classifier import classify, estimate_risk
from .escalation import EscalationController
from .model_select import plan_tiers
from .remote import FireworksClient, Ledger
from .schemas import Route, Solved, TaskSpec

log = logging.getLogger("gemmagate.router")

try:  # optional local model (free tokens); absent by default in the image
    from .local_model import load_local_model
except Exception:  # pragma: no cover
    load_local_model = None


class Router:
    def __init__(self, max_workers: int = 6, max_remote_calls: int = 3):
        allowed = [m for m in os.environ.get("ALLOWED_MODELS", "").split(",")
                   if m.strip()]
        overrides = _tier_overrides_from_env()
        if allowed:
            self.tiers = plan_tiers(allowed, overrides).as_dict()
        else:
            log.warning("ALLOWED_MODELS empty — remote tiers disabled")
            self.tiers = {}
        self.ledger = Ledger()
        self.client = FireworksClient(self.ledger)
        self.max_workers = max_workers
        self.local_model = None
        if load_local_model is not None:
            try:
                self.local_model = load_local_model()
            except Exception as e:
                log.info("local model unavailable (%s); heuristics only", e)
        self.controller = EscalationController(
            self.client, self.tiers, max_remote_calls=max_remote_calls,
            local_model=self.local_model)
        if self.local_model is not None:
            try:                               # measure real tokens/sec once
                t0 = time.time()
                probe = self.local_model.generate("Count: 1 2 3 4 5 6 7 8",
                                                  max_tokens=16, temperature=0.0)
                dt = max(time.time() - t0, 1e-3)
                self.controller.local_tps = max(probe.output_tokens, 1) / dt
                log.info("local model speed: %.1f tok/s",
                         self.controller.local_tps)
            except Exception:
                pass
        self.batcher = SentimentBatcher(self.client, self.tiers.get("cheap"))
        self.fact_batcher = ShortAnswerBatcher(self.client, self.tiers.get("cheap"))

    # ------------------------------------------------------------- solve

    def solve_all(self, tasks: list[dict], deadline: float) -> list[Solved]:
        specs: list[TaskSpec] = []
        for t in tasks:
            try:
                spec = classify(t["task_id"], t["prompt"], self.local_model)
                estimate_risk(spec)
            except Exception as e:
                log.error("classification failed for %s: %s", t.get("task_id"), e)
                spec = TaskSpec(task_id=str(t.get("task_id", "?")),
                                prompt=str(t.get("prompt", "")))
            specs.append(spec)
            log.info("%s -> %s / %s", spec.task_id, spec.category.value,
                     spec.risk.value)

        # key results by INPUT INDEX, not task_id — duplicate ids with
        # different prompts must each get their own answer
        results: dict[int, Solved] = {}

        # ---- sentiment batch pre-pass: lexicon first (free), then one
        # batched remote call for the residues; per-item validation failures
        # fall through to the normal individual ladder below.
        from .schemas import Category
        residues: list[tuple[int, Solved]] = []
        batch_candidates: list[tuple[int, TaskSpec]] = []
        for i, s in enumerate(specs):
            if s.category == Category.SENTIMENT and self.batcher.eligible(s):
                local = self.controller._local(s)
                if local is not None and local.validation.passed:
                    results[i] = self.controller._done(s, local, [local],
                                                       time.time())
                else:
                    batch_candidates.append((i, s))
        if len(batch_candidates) >= 2:
            results.update(self.batcher.solve(batch_candidates, deadline))
        fact_candidates = [(i, s) for i, s in enumerate(specs)
                           if i not in results and s.category == Category.FACTUAL
                           and s.cls_confidence >= 0.6 and not s.payload]
        if len(fact_candidates) >= 2:
            results.update(self.fact_batcher.solve(fact_candidates, deadline))

        pending = [(i, s) for i, s in enumerate(specs) if i not in results]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._solve_one, s, deadline): i
                       for i, s in pending}
            for fut in as_completed(futures):
                i = futures[fut]
                s = specs[i]
                try:
                    results[i] = fut.result()
                except Exception as e:  # absolute per-task isolation
                    log.error("%s: solver raised %s", s.task_id, e)
                    results[i] = Solved(
                        task_id=s.task_id, route=Route.FAILSAFE,
                        answer="Unable to determine a reliable answer.",
                        category=s.category, risk=s.risk)
        return [results[i] for i in range(len(specs)) if i in results]

    def _solve_one(self, spec: TaskSpec, deadline: float) -> Solved:
        solved = self.controller.solve(spec, deadline)
        if solved.remote_tokens == 0 and solved.route.value.startswith("local"):
            solved.answer = polish(spec, solved.answer)
        log.info("%s: route=%s conf=%.2f remote_tokens=%d t=%.1fs",
                 spec.task_id, solved.route.value, solved.confidence,
                 solved.remote_tokens, solved.wall_time_s)
        return solved


def _tier_overrides_from_env() -> dict:
    """Optional launch-day pins without code changes:
       GEMMAGATE_TIER_CHEAP / _MID / _STRONG = substring of a model name."""
    out = {}
    for tier in ("cheap", "mid", "strong"):
        v = os.environ.get(f"GEMMAGATE_TIER_{tier.upper()}")
        if v:
            out[tier] = v
    return out
