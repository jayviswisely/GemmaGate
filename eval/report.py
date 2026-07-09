#!/usr/bin/env python3
"""Per-question report: every prompt, the answer given, expected, verdict.

Usage:
  python eval/report.py                                   # original set, dry
  python eval/report.py --tasks eval/gold_tasks_hard.json
  python eval/report.py --tasks eval/gold_tasks_extreme.json --real
Writes report.txt next to the console output.

Verdicts:
  CORRECT    — answer matches the check
  WRONG      — answer fails the check (in dry-run only shown for LOCAL routes)
  ESCALATED* — dry-run remote answer is simulated, so correctness is unknowable
"""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
p = argparse.ArgumentParser()
p.add_argument("--tasks", default=os.path.join(os.path.dirname(__file__), "gold_tasks.json"))
p.add_argument("--real", action="store_true")
p.add_argument("--out", default="report.txt")
a = p.parse_args()

if not a.real:
    os.environ.setdefault("GEMMAGATE_DRY_RUN", "1")
    os.environ.setdefault("ALLOWED_MODELS", "sim-8b,sim-70b")
    os.environ.setdefault("FIREWORKS_API_KEY", "sim")
    os.environ.setdefault("FIREWORKS_BASE_URL", "https://sim.invalid/v1")

from run_benchmark import score          # noqa: E402
from gemmagate.router import Router      # noqa: E402

tasks = json.load(open(a.tasks, encoding="utf-8"))
router = Router()
solved = {s.task_id: s for s in router.solve_all(
    [{"task_id": t["task_id"], "prompt": t["prompt"]} for t in tasks],
    time.time() + 480)}

lines, n_ok, n_wrong, n_esc = [], 0, 0, 0
for t in tasks:
    s = solved[t["task_id"]]
    local = s.remote_tokens == 0 and s.route.value.startswith("local")
    ok = score(t["check"], s.answer)
    if not a.real and not local:
        verdict, n_esc = "ESCALATED*", n_esc + 1
    elif ok:
        verdict, n_ok = "CORRECT", n_ok + 1
    else:
        verdict, n_wrong = "WRONG", n_wrong + 1
    lines += [f"[{verdict:<10}] {t['task_id']}  route={s.route.value}  tokens={s.remote_tokens}",
              f"  Q: {t['prompt'][:300]}",
              f"  A: {(s.answer or '')[:300]}",
              f"  EXPECTED: {json.dumps(t['check'], ensure_ascii=False)[:220]}", ""]

hdr = (f"{os.path.basename(a.tasks)}  mode={'REAL' if a.real else 'DRY'}  "
       f"CORRECT={n_ok}  WRONG={n_wrong}  ESCALATED*={n_esc}"
       + ("" if a.real else "   (*simulated answers — not scoreable in dry-run)"))
out = hdr + "\n" + "=" * len(hdr) + "\n" + "\n".join(lines)
print(out)
open(a.out, "w", encoding="utf-8").write(out)
print(f"\nsaved -> {a.out}")
