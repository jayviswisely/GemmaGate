#!/usr/bin/env python3
"""Container entrypoint — the crash barrier.

Guarantees regardless of what happens inside:
  * /output/results.json is written, is valid JSON, covers every task_id
  * every answer is a non-empty string (safe fallback if a task was missed)
  * exit code 0 whenever the output file was written successfully
  * a hard wall-clock watchdog: even if the router HANGS (stuck socket,
    pathological task), the final write still happens inside the budget
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time

logging.basicConfig(
    stream=sys.stderr, level=os.environ.get("GEMMAGATE_LOG", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("gemmagate.main")

TIME_BUDGET_S = float(os.environ.get("GEMMAGATE_TIME_BUDGET", "570"))  # 9.5 min
WATCHDOG_GRACE_S = 15.0          # reserved for the final write + exit
SAFE_FALLBACK = "Unable to determine a reliable answer."


def main() -> int:
    start = time.time()
    deadline = start + TIME_BUDGET_S

    from gemmagate.io_layer import load_tasks, write_results
    tasks = load_tasks()
    log.info("loaded %d tasks", len(tasks))

    # Placeholder file immediately: output exists even under OOM/kill.
    results = [{"task_id": t["task_id"], "answer": SAFE_FALLBACK} for t in tasks]
    write_results(results)

    # Solve in a watchdog thread. `box` is the thread's only output channel;
    # if the router hangs past the budget we proceed without it.
    box: dict = {"solved": None, "ledger": None, "error": None}

    def _work():
        try:
            from gemmagate.router import Router
            router = Router()
            box["solved"] = router.solve_all(tasks, deadline)
            box["ledger"] = router.ledger.snapshot()
        except Exception as e:                      # noqa: BLE001
            box["error"] = e

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    worker.join(timeout=max(5.0, deadline - time.time() - WATCHDOG_GRACE_S))

    if worker.is_alive():
        log.critical("router exceeded the time budget — writing best-effort "
                     "results and exiting (watchdog)")
    elif box["error"] is not None:
        log.critical("router failed: %s — writing best-effort results",
                     box["error"], exc_info=box["error"])

    solved = box["solved"]
    if solved is not None:
        # Router returns one result per input, IN INPUT ORDER — map by
        # position, not task_id, so duplicate ids each keep their own answer.
        if len(solved) == len(tasks):
            results = [{"task_id": t["task_id"],
                        "answer": (s.answer or "").strip() or SAFE_FALLBACK}
                       for t, s in zip(tasks, solved)]
        else:                                       # defensive: partial batch
            by_id = {}
            for s in solved:
                by_id.setdefault(s.task_id, s)
            results = [{"task_id": t["task_id"],
                        "answer": ((by_id[t["task_id"]].answer or "").strip()
                                   or SAFE_FALLBACK)
                        if t["task_id"] in by_id else SAFE_FALLBACK}
                       for t in tasks]
    if box["ledger"]:
        log.info("token ledger: %s", json.dumps(box["ledger"]))

    ok = write_results(results)
    log.info("done in %.1fs, output_ok=%s", time.time() - start, ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
