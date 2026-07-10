"""Fireworks AI client.

Contract compliance:
  * base URL from FIREWORKS_BASE_URL env only (path joined at runtime)
  * API key from FIREWORKS_API_KEY env only
  * model IDs are passed in by the caller — nothing hardcoded here

Real token usage from the API's `usage` field feeds the ledger; local
estimates are used only when the field is absent. GEMMAGATE_DRY_RUN=1 swaps
in a simulator so the whole agent can be tested without network access.

Fireworks reasoning models may bill hidden reasoning inside completion_tokens.
By default the client sends reasoning_effort="none"; if a proxy rejects that
field with HTTP 400, it retries without the field and drops it for the rest of
the run.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

from .schemas import LLMResult

log = logging.getLogger("gemmagate.remote")

_SYSTEM = ("You are a precise task solver. Output only what is asked, in the "
           "exact format requested. No preamble, no extra commentary.")


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.7)) if text else 0


class Ledger:
    def __init__(self):
        self._lock = threading.Lock()
        self.input = 0
        self.output = 0
        self.calls = 0
        self.per_model: dict[str, dict] = {}

    def record(self, model: str, i: int, o: int):
        with self._lock:
            self.input += i
            self.output += o
            self.calls += 1
            m = self.per_model.setdefault(model, {"input": 0, "output": 0, "calls": 0})
            m["input"] += i; m["output"] += o; m["calls"] += 1

    @property
    def total(self) -> int:
        return self.input + self.output

    def snapshot(self) -> dict:
        with self._lock:
            return {"remote_input": self.input, "remote_output": self.output,
                    "remote_total": self.total, "calls": self.calls,
                    "per_model": {k: dict(v) for k, v in self.per_model.items()}}


class FireworksClient:
    def __init__(self, ledger: Optional[Ledger] = None,
                 max_retries: int = 2, timeout: float = 45.0):
        self.ledger = ledger or Ledger()
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        base = os.environ.get("FIREWORKS_BASE_URL", "").rstrip("/")
        # Accept either a bare base URL or one already ending in the endpoint
        if base.endswith("/chat/completions"):
            self.url = base
        elif base:
            self.url = base + "/chat/completions"
        else:
            self.url = ""
        self.max_retries = max_retries
        self.timeout = timeout
        self.dry_run = os.environ.get("GEMMAGATE_DRY_RUN", "") == "1" or not self.url
        self._dry_responses: dict[str, str] = {}
        self.reasoning_off = os.environ.get("GEMMAGATE_REASONING_OFF", "1") != "0"
        self._drop_reasoning_param = False

    # ------------------------------------------------------------- public

    def complete(self, model: str, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.0, json_mode: bool = False) -> LLMResult:
        if self.dry_run:
            return self._simulate(model, prompt, max_tokens)

        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
        }
        if self.reasoning_off and not self._drop_reasoning_param:
            body["reasoning_effort"] = "none"
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        data, err = None, None
        t0 = time.time()
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.url, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json",
                             "Accept": "application/json",
                             "User-Agent": "GemmaGate/1.0",
                             "Authorization": f"Bearer {self.api_key}"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode(errors="replace")[:300]
                except Exception:
                    detail = ""
                err = f"HTTP {e.code} {detail}".strip()
                if json_mode and e.code == 400 and "response_format" in body:
                    # some models reject response_format — retry without it
                    body.pop("response_format", None)
                    json_mode = False
                    continue
                if e.code == 400 and "reasoning_effort" in body:
                    body.pop("reasoning_effort", None)
                    self._drop_reasoning_param = True
                    log.info("remote endpoint rejected reasoning_effort; "
                             "dropping it for the rest of this run")
                    continue
                if e.code in (429, 500, 502, 503) and attempt < self.max_retries:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    time.sleep(0.8)
                    continue
                break

        if data is None:
            log.warning("remote call failed (%s) model=%s", err, model)
            return LLMResult(text="", model=model, is_remote=True, error=err)

        text = ""
        try:
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError):
            pass
        usage = data.get("usage") or {}
        i = int(usage.get("prompt_tokens", estimate_tokens(prompt)))
        o = int(usage.get("completion_tokens", estimate_tokens(text)))
        self.ledger.record(model, i, o)
        log.info("remote model=%s in=%d out=%d t=%.1fs", model, i, o, time.time() - t0)
        return LLMResult(text=text, input_tokens=i, output_tokens=o,
                         model=model, is_remote=True)

    # ------------------------------------------------------------ dry run

    def set_dry_responses(self, mapping: dict[str, str]):
        self._dry_responses = mapping

    def _simulate(self, model: str, prompt: str, max_tokens: int) -> LLMResult:
        text = "DRY_RUN"
        for key, val in self._dry_responses.items():
            if key.lower() in prompt.lower():
                text = val
                break
        i, o = estimate_tokens(prompt), estimate_tokens(text)
        self.ledger.record(model + "(dry)", i, o)
        return LLMResult(text=text, input_tokens=i, output_tokens=o,
                         model=model + "(dry)", is_remote=True)
