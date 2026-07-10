"""Local model layer (free tokens) — drafting, classification fallback.

Backends, chosen by env at startup:
  * GEMMAGATE_LOCAL_GGUF=<path.gguf>   -> llama-cpp-python (CPU, recommended:
        qwen2.5-3b-instruct Q4_K_M for the optional Dockerfile.local profile)
  * GEMMAGATE_LOCAL_MODEL=<hf id/path> -> transformers
  * neither set                        -> None (agent runs heuristics-only)

The agent is fully functional without a local model; with one, the
draft-conditional remote rung activates (see escalation.py) and converts
long remote outputs into 1-2 token "OK" verifications.
"""
from __future__ import annotations

import logging
import os
import threading

from .remote import estimate_tokens
from .schemas import LLMResult

log = logging.getLogger("gemmagate.local")


class StubLocal:
    """Deterministic local model for tests: substring-of-prompt -> reply."""

    def __init__(self, responses: dict | None = None):
        self.responses = responses or {}

    def generate(self, prompt: str, max_tokens: int = 64,
                 temperature: float = 0.0) -> LLMResult:
        text = ""
        for k, v in self.responses.items():
            if k.lower() in prompt.lower():
                if isinstance(v, list):          # sequential answers
                    text = v.pop(0) if v else ""
                else:
                    text = v
                break
        return LLMResult(text=text, input_tokens=estimate_tokens(prompt),
                         output_tokens=estimate_tokens(text),
                         model="stub-local", is_remote=False)


class _LlamaCppModel:
    def __init__(self, path: str):
        from llama_cpp import Llama
        threads = int(os.environ.get(
            "GEMMAGATE_LOCAL_THREADS",
            str(max(1, min(os.cpu_count() or 2, 4)))))
        self.llm = Llama(model_path=path, n_ctx=4096,
                         n_threads=threads, verbose=False)
        self._lock = threading.Lock()

    def generate(self, prompt: str, max_tokens: int = 64,
                 temperature: float = 0.0) -> LLMResult:
        with self._lock:
            out = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature)
        text = (out["choices"][0]["message"]["content"] or "").strip()
        u = out.get("usage", {})
        return LLMResult(text=text,
                         input_tokens=u.get("prompt_tokens", estimate_tokens(prompt)),
                         output_tokens=u.get("completion_tokens", estimate_tokens(text)),
                         model="local-gguf", is_remote=False)


class _TransformersModel:
    def __init__(self, path: str):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model=path, device_map="auto")
        self._lock = threading.Lock()

    def generate(self, prompt: str, max_tokens: int = 64,
                 temperature: float = 0.0) -> LLMResult:
        with self._lock:
            out = self.pipe([{"role": "user", "content": prompt}],
                            max_new_tokens=max_tokens,
                            do_sample=temperature > 0,
                            temperature=max(temperature, 1e-3),
                            return_full_text=False)
        text = out[0]["generated_text"]
        if isinstance(text, list):
            text = text[-1].get("content", "")
        return LLMResult(text=(text or "").strip(),
                         input_tokens=estimate_tokens(prompt),
                         output_tokens=estimate_tokens(text or ""),
                         model="local-hf", is_remote=False)


def load_local_model():
    gguf = os.environ.get("GEMMAGATE_LOCAL_GGUF")
    if gguf and os.path.exists(gguf):
        m = _LlamaCppModel(gguf)
        log.info("local GGUF model loaded: %s", gguf)
        return m
    path = os.environ.get("GEMMAGATE_LOCAL_MODEL")
    if path:
        m = _TransformersModel(path)
        log.info("local transformers model loaded: %s", path)
        return m
    return None
