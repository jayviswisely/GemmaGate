"""I/O layer — the harness contract lives here.

Contract (non-negotiable):
  * read  /input/tasks.json   (list of {"task_id", "prompt"})
  * write /output/results.json (list of {"task_id", "answer"}) — ALWAYS valid
    JSON, ALWAYS containing every input task_id, written atomically.

Paths are overridable via INPUT_PATH / OUTPUT_PATH env vars for local testing.
Malformed input is handled defensively: entries missing fields are kept with
best-effort coercion so no task_id is ever dropped from the output.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile

log = logging.getLogger("gemmagate.io")

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")


def load_tasks() -> list[dict]:
    """Return a list of {"task_id": str, "prompt": str}. Never raises."""
    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        log.error("input file missing: %s", INPUT_PATH)
        return []
    except json.JSONDecodeError as e:
        log.error("input is not valid JSON: %s", e)
        return _salvage_task_ids()
    except Exception as e:  # pragma: no cover
        log.error("unexpected input error: %s", e)
        return []

    if isinstance(data, dict):  # tolerate {"tasks": [...]}
        data = data.get("tasks", [])
    if not isinstance(data, list):
        log.error("input root is not a list")
        return []

    tasks: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            log.warning("entry %d is not an object; skipped", i)
            continue
        task_id = item.get("task_id")
        prompt = item.get("prompt")
        if task_id is None:
            task_id = f"missing_id_{i}"
            log.warning("entry %d missing task_id; assigned %s", i, task_id)
        if not isinstance(prompt, str):
            prompt = "" if prompt is None else str(prompt)
            log.warning("entry %d has non-string prompt; coerced", i)
        tasks.append({"task_id": str(task_id), "prompt": prompt})
    return tasks


def _salvage_task_ids() -> list[dict]:
    """If the JSON is broken, regex-salvage task_ids so results.json can
    still cover them (empty answers beat missing task_ids)."""
    import re
    try:
        raw = open(INPUT_PATH, "r", encoding="utf-8", errors="replace").read()
        ids = re.findall(r'"task_id"\s*:\s*"([^"]+)"', raw)
        return [{"task_id": tid, "prompt": ""} for tid in ids]
    except Exception:
        return []


def write_results(results: list[dict]) -> bool:
    """Atomic write of results.json. Returns True on success. Never raises."""
    # Sanitize: every entry must be {"task_id": str, "answer": str}
    clean = []
    for r in results:
        try:
            clean.append({"task_id": str(r["task_id"]),
                          "answer": "" if r.get("answer") is None else str(r["answer"])})
        except Exception:
            continue
    try:
        out_dir = os.path.dirname(OUTPUT_PATH) or "."
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=1)
        os.replace(tmp, OUTPUT_PATH)  # atomic on POSIX
        return True
    except Exception as e:
        log.error("primary write failed: %s", e)
        try:  # last resort: direct minimal write
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                f.write(json.dumps(clean))
            return True
        except Exception as e2:
            log.critical("output write failed entirely: %s", e2)
            return False
