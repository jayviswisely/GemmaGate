#!/usr/bin/env python3
"""GemmaGate benchmark harness — the tool that answers "does it actually use
the least tokens, at passing accuracy?"

It runs the SAME gold-labeled task set through two agents on identical
infrastructure and prints a head-to-head table:

  * GemmaGate  — the full routing agent
  * Baseline   — what a naive submission does: every task sent raw to the
                 strongest allowed model, max_tokens=512

and scores every answer automatically (numeric compare, exact label,
JSON-subset for NER, sentence-limit+keyword for summaries, and — for code —
actually EXECUTING the returned function against test cases).

Usage:
  # offline sanity (routing + token estimates; remote answers are simulated,
  # so remote-routed accuracy is meaningless in this mode):
  python3 eval/run_benchmark.py

  # the real judgment (uses your Fireworks key, spends real tokens — the
  # gold set costs roughly 2-5k baseline tokens total):
  export FIREWORKS_API_KEY=fw_...
  export FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
  export ALLOWED_MODELS="<cheap>,<mid>,<strong>"
  python3 eval/run_benchmark.py --real

How to read the verdict:
  1. accuracy gate first: GemmaGate accuracy must be >= baseline (or at
     least >= the competition threshold);
  2. only then compare total remote tokens — that ratio is your score story.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ------------------------------------------------------------------ scoring


def _extract_number(text: str):
    m = re.search(r"ANSWER\s*[:=]\s*(.+)", text, re.I)
    if m:
        text = m.group(1)
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace("$", ""))
    return float(nums[-1].replace(",", "")) if nums else None


def _run_code(code: str, tests: list[str]) -> bool:
    code = re.sub(r"^```(?:\w+)?\n|```\s*$", "", code.strip(), flags=re.M)
    ns: dict = {}
    try:
        exec(code, ns)  # noqa: S102 — trusted eval-set code, sandboxed run advised for foreign sets
        for t in tests:
            exec(t, ns)  # noqa: S102
        return True
    except Exception:
        return False


def score(check: dict, answer: str) -> bool:
    mode = check["mode"]
    a = (answer or "").strip()
    if mode == "number":
        v = _extract_number(a)
        return v is not None and abs(v - float(check["expected"])) < 1e-6
    if mode == "exact":
        head = re.split(r"[—\-:.]", a, maxsplit=1)[0].strip().lower()
        exp = str(check["expected"]).lower()
        return head == exp or a.lower() == exp or \
            re.search(r"\b" + re.escape(exp) + r"\b", a, re.I) is not None
    if mode == "contains_all":
        return all(k.lower() in a.lower() for k in check["expected"])
    if mode == "json_subset":
        try:
            m = re.search(r"\{.*\}", a, re.S)
            obj = json.loads(m.group(0) if m else a)
        except Exception:
            return False
        for key, wanted in check["expected"].items():
            got = [str(x).lower() for x in obj.get(key, [])]
            if not all(any(w.lower() in g or g in w.lower() for g in got)
                       for w in wanted):
                return False
        return True
    if mode == "summary":
        sents = [s for s in re.split(r"(?<=[.!?])\s+", a) if s.strip()]
        if len(sents) > check.get("max_sentences", 99):
            return False
        return any(k.lower() in a.lower() for k in check.get("contains_any", []))
    if mode == "code_test":
        return _run_code(a, check["tests"])
    raise ValueError(f"unknown check mode {mode}")


# ------------------------------------------------------------------ agents


def run_gemmagate(tasks: list[dict]):
    from gemmagate.router import Router
    router = Router()
    deadline = time.time() + 540
    solved = router.solve_all(
        [{"task_id": t["task_id"], "prompt": t["prompt"]} for t in tasks], deadline)
    answers = {s.task_id: s for s in solved}
    return answers, router.ledger.snapshot()


def run_baseline(tasks: list[dict]):
    """The naive submission: raw prompt -> strongest model, generous cap."""
    from gemmagate.model_select import plan_tiers
    from gemmagate.remote import FireworksClient, Ledger
    allowed = [m for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
    strong = plan_tiers(allowed).strong
    ledger = Ledger()
    client = FireworksClient(ledger)
    answers = {}
    for t in tasks:
        res = client.complete(strong, t["prompt"], max_tokens=512, temperature=0.0)
        answers[t["task_id"]] = res.text
    return answers, ledger.snapshot()


# ------------------------------------------------------------------ report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="use the real Fireworks API (default: dry-run)")
    ap.add_argument("--tasks", default=os.path.join(os.path.dirname(__file__),
                                                    "gold_tasks.json"))
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    if not args.real:
        os.environ["GEMMAGATE_DRY_RUN"] = "1"
        os.environ.setdefault("ALLOWED_MODELS", "sim-8b-instruct,sim-34b,sim-70b-instruct")
        os.environ.setdefault("FIREWORKS_API_KEY", "sim")
        os.environ.setdefault("FIREWORKS_BASE_URL", "https://sim.invalid/v1")
        print("MODE: DRY-RUN — token numbers are estimates; accuracy is only "
              "meaningful for locally-solved tasks (remote answers are simulated).\n")
    else:
        os.environ.pop("GEMMAGATE_DRY_RUN", None)
        for var in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"):
            if not os.environ.get(var):
                print(f"--real requires {var} to be set"); return 1
        print("MODE: REAL API — spending real tokens.\n")

    tasks = json.load(open(args.tasks))

    gg_answers, gg_ledger = run_gemmagate(tasks)
    if not args.skip_baseline:
        bl_answers, bl_ledger = run_baseline(tasks)
    else:
        bl_answers, bl_ledger = {}, {"remote_total": 0}

    # ---- table
    W = 10
    print(f"{'task':<9}{'category':<26}{'route':<14}"
          f"{'gg_tok':>{W}}{'gg_ok':>7}{'bl_ok':>7}")
    print("-" * 73)
    gg_correct = bl_correct = 0
    local_count = 0
    for t in tasks:
        tid = t["task_id"]
        s = gg_answers[tid]
        ok = score(t["check"], s.answer)
        gg_correct += ok
        local = s.route.value in ("local_rule", "local_model", "failsafe") \
            and s.remote_tokens == 0
        local_count += local
        b_ok = ""
        if bl_answers:
            b = score(t["check"], bl_answers.get(tid, ""))
            bl_correct += b
            b_ok = "Y" if b else "n"
        print(f"{tid:<9}{s.category.value:<26}{s.route.value:<14}"
              f"{s.remote_tokens:>{W}}{'Y' if ok else 'n':>7}{b_ok:>7}")

    n = len(tasks)
    print("-" * 73)
    print(f"\n{'':<26}{'GemmaGate':>14}{'Baseline':>14}")
    print(f"{'accuracy':<26}{gg_correct}/{n:<12}"
          f"{(str(bl_correct) + '/' + str(n)) if bl_answers else '—':>14}")
    print(f"{'total remote tokens':<26}{gg_ledger['remote_total']:>14}"
          f"{bl_ledger['remote_total']:>14}")
    print(f"{'remote calls':<26}{gg_ledger.get('calls', 0):>14}"
          f"{bl_ledger.get('calls', 0):>14}")
    print(f"{'solved fully locally':<26}{local_count}/{n:<12}{'0/' + str(n):>13}")
    if bl_ledger["remote_total"]:
        saved = 1 - gg_ledger["remote_total"] / bl_ledger["remote_total"]
        print(f"{'token reduction':<26}{saved:>13.0%}")
    print("\nVERDICT ORDER: (1) GemmaGate accuracy must be >= baseline / "
          "threshold. (2) Only then does the token reduction count.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
