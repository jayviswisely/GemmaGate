# GemmaGate — Testing Strategy

## Principle
Two failure classes lose this competition: breaking the harness contract (instant zero) and shipping wrong answers from over-eager local solvers (fails the accuracy gate). The test strategy attacks both, fully offline, so every commit can be verified in seconds without an API key.

## Layer 1 — Harness-contract tests (subprocess-level)
Run `main.py` as a real subprocess with `INPUT_PATH`/`OUTPUT_PATH` redirected to temp dirs, asserting on exit codes and file contents exactly as the grader would:

- **Coverage invariant**: every input `task_id` appears in `results.json`, even when prompts are missing or null.
- **Malformed-input survival**: syntactically broken `tasks.json` still yields exit 0 and a valid `results.json` with regex-salvaged task_ids.
- Additional manual probes worth running before submission: empty file, empty list, huge single prompt, duplicate task_ids, read-only `/output` simulation.

## Layer 2 — Routing-invariant tests (in-process, dry-run remote)
`GEMMAGATE_DRY_RUN=1` swaps the Fireworks client for a simulator that returns canned answers while still charging estimated tokens to the ledger, so token assertions stay meaningful:

- Deterministic math (`19*21`, `15% of 80`, price-increase phrasing) ⇒ correct answer AND `remote_tokens == 0`.
- Sentiment gating: unambiguous review answered locally; mixed review returns `None` (escalates) rather than guessing.
- NER coverage guard: fully-covered text answered locally; text with unknown capitalized spans escalates.
- Logic brute-force: provable puzzle answered locally; a *remote* answer contradicting the brute-force solution is rejected and repaired with the proven value.
- Code: local `=`→`==` fix validated; unchanged buggy code rejected; `ast.parse` gate.
- Summary length constraints enforced by the validator.
- Failsafe: with no remote tiers at all, answers are still non-empty.

## Layer 3 — Precision regression for local solvers
The most dangerous bug class is a local solver answering a task it shouldn't (v1 lesson: a math regex read "voted 7-2" inside a passage as arithmetic). Defenses under test:
- category gating (solvers only run for their classified category),
- evaluate-and-verify expression finding (must contain an operator AND evaluate),
- confidence gates (sentiment margin, NER coverage, logic uniqueness).
When adding lexicon words, gazetteer entries, or regex patterns, add a negative test alongside every positive one.

## Layer 4 — Pre-launch dress rehearsal
1. Build the Docker image; run against `examples/tasks.json` with real env vars and a real Fireworks key; verify ledger totals in stderr.
2. Timebox test: duplicate the example set to 200+ tasks and confirm completion within budget, with deadline-pressure failsafes firing if artificially shortened (`GEMMAGATE_TIME_BUDGET=30`).
3. Tier sanity: run with the revealed `ALLOWED_MODELS` string and read the `model tiers:` log line; pin with `GEMMAGATE_TIER_*` overrides if the heuristic misranks anything.
