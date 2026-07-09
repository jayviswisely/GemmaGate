# GemmaGate — Cost-Aware Hybrid Inference Router

Competition submission for the **Hybrid Token-Efficient Routing Agent** track. GemmaGate reads `/input/tasks.json`, solves each task through a validation-guided, local-first escalation ladder, and writes `/output/results.json` — spending Fireworks tokens only when a free local check *proves* the free answer isn't good enough.

## Build & run

```bash
# build (stdlib-only runtime => image is ~150MB, far under the 10GB cap)
docker build -t gemmagate .

# run exactly as the harness does
docker run --rm \
  -v "$PWD/examples:/input:ro" \
  -v "$PWD/out:/output" \
  -e FIREWORKS_API_KEY=fw_... \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS="model-a-8b-instruct,model-b-70b-instruct" \
  gemmagate

cat out/results.json
```

Local development without Docker or network:

```bash
GEMMAGATE_DRY_RUN=1 \
ALLOWED_MODELS="test-8b,test-70b" FIREWORKS_API_KEY=x FIREWORKS_BASE_URL=https://x \
INPUT_PATH=examples/tasks.json OUTPUT_PATH=/tmp/results.json \
python3 main.py

python3 tests/test_all.py        # 11/11 offline tests
```

## How it works

```
main.py  ── crash barrier: placeholder results.json written FIRST, exit 0 guaranteed
  │
  ├─ io_layer      read/validate tasks.json · salvage task_ids from broken JSON
  │                · atomic results.json write · every task_id always present
  │
  ├─ router        tier ALLOWED_MODELS by name hints · ThreadPool (6 workers)
  │                · per-task exception isolation · deadline manager (9.5 min budget)
  │
  └─ per task:  classify (8 categories, regex heuristics; remote NEVER used)
                → risk (low/med/high)
                → escalation ladder, each rung gated by the FREE validator:
                     local rule solver → cheap remote → mid → strong
                → failsafe answer if everything fails (never an empty string)
```

### Escalation ladders by category

| Category | Ladder | Local rung |
|---|---|---|
| Math | rules → cheap → strong | AST-whitelisted safe eval, percentages, price changes, aggregates, unit rates, ratios, item totals, profit, percent change, compound growth, simple interest |
| Sentiment | lexicon → cheap | negation-aware lexicon with contrast-clause weighting (post-"but" verdict) and sarcasm abstention; custom label sets mapped via synonyms |
| NER | rules → cheap → mid | regex dates/orgs/honorifics + gazetteer, with a **coverage guard**: any unlabeled capitalized span ⇒ escalate |
| Summarization | extractive → cheap → mid | frequency-scored sentence selection, only when the length constraint is checkable |
| Logic | brute force → cheap → strong | permutation search over parsed comparatives — answers are **provably correct** when returned |
| Factual | cheap → mid (→ strong) | never guessed locally |
| Code debug | mechanical fixes → **mid** → strong | `=`-in-condition fix only; cheap tier skipped (expected-value routing) |
| Code gen | **mid** → strong | syntax-validated locally via `ast.parse` |

### Local model × Fireworks: the two compute-combining rungs

**Draft-conditional generation.** With a local model baked in (Dockerfile profile B), factual / summarization / logic tasks get a free local draft that is validated locally, then attached to the single remote call with the contract *"reply exactly OK if PROPOSED is correct, otherwise output the corrected answer only."* Draft accepted → remote output is 1–2 tokens instead of 100–180. Draft wrong → the same call returns the correction; no second call, no resent payload. EV-positive whenever `P(draft ok) × output_tokens > draft_tokens`.

**Sentiment batching.** Residues the lexicon abstained on are grouped by identical label sets and sent as one numbered call (≤6/chunk, ~4 output tokens per item), amortizing the instruction header. Every line is validated per-task; anything unparsed or invalid returns to the individual ladder, so batching can lower tokens but never accuracy.

### Token-efficiency levers

1. **In-run exact-duplicate memoization.** Byte-identical prompts within one invocation are solved once (single-flight); nothing is persisted and no fuzzy matching is used, keeping clear of the no-answer-caching rule while never paying twice for the same call.
1. **Local-first with validation gates.** ~60–70% of benchmark-style tasks (math, clear sentiment, covered NER, constrained summaries, ordering logic) finish at 0 remote tokens; the validator (free, deterministic) is what authorizes stopping.
2. **Compact category prompts + minimal-output caps.** Sentiment label = 8 max_tokens; logic answer = 120; JSON mode for NER. No chain-of-thought requested — *except* math, where brief working ending in `ANSWER: <value>` is allowed because small models are far more accurate with it and the accuracy gate ranks first; the validator extracts just the value.
3. **Repair prompts on retry.** A failed remote answer is retried with only the bad output + violated constraint (~30–120 tokens), never a full resend.
4. **Cross-validation catches wrong remote answers free.** Math answers are re-derived by the local solver; logic answers are checked against the brute-force solution — if they conflict, the *validator supplies the provably correct value* as the repair.

### Harness-contract hardening

- Placeholder `results.json` is written **before** solving starts — output exists even under OOM/kill.
- Atomic writes (`os.replace`), task_ids regex-salvaged from syntactically broken input, non-string prompts coerced, missing ids synthesized.
- Per-task try/except in the thread pool: one pathological task can't sink the batch. Results are keyed by input index, so duplicate task_ids with different prompts each get their own answer.
- Classification confidence: weighted rule voting; a shaky category assignment automatically raises risk so the escalation ladder compensates.
- Deadline manager: remote rungs are skipped in the final 20s; pending tasks get local best-effort/failsafe answers.
- Exit code 0 whenever the output file was written.

## How to judge it: the benchmark harness

```bash
# offline: verifies ROUTING (what stays local) and token estimates
python3 eval/run_benchmark.py

# the real verdict: same gold set, real Fireworks key (~2-5k baseline tokens)
export FIREWORKS_API_KEY=fw_... FIREWORKS_BASE_URL=... ALLOWED_MODELS=...
python3 eval/run_benchmark.py --real
```

It runs the gold-labeled set (`eval/gold_tasks.json`, all 8 categories) through GemmaGate AND a naive baseline (raw prompt → strongest model), scores every answer automatically — numeric compare, exact labels, JSON-subset for NER, sentence-limit for summaries, and **executing returned code against test cases** — and prints accuracy + total tokens side by side. Verdict order: accuracy gate first, then the token reduction. Extend `gold_tasks.json` with your own labeled tasks to harden the estimate.

## Launch-day configuration (no code changes)

Tiers are computed from `ALLOWED_MODELS` automatically (parameter-count parsing `8b`/`70b`/`8x7b`, keywords `mini`/`large`, instruct-variant preference; unknown names default to mid-size — never accidentally "cheap"). To pin tiers manually:

```bash
-e GEMMAGATE_TIER_CHEAP=8b -e GEMMAGATE_TIER_STRONG=70b   # substring match
```

Other knobs: `GEMMAGATE_TIME_BUDGET` (seconds, default 570), `GEMMAGATE_LOG` (level), `GEMMAGATE_LOCAL_MODEL` (path/HF id to enable the optional local-model classification fallback — see Dockerfile comments for baking weights).

## Repository layout

```
main.py                    entrypoint / crash barrier
gemmagate/
  io_layer.py              harness I/O contract
  classifier.py            8-category heuristics + structure extraction
  model_select.py          ALLOWED_MODELS -> cheap/mid/strong tiers
  remote.py                Fireworks client (FIREWORKS_BASE_URL), ledger, dry-run
  prompts.py               category templates, compression, repair prompts
  validator.py             per-category validation + auto-repair
  escalation.py            ladder controller, deadline logic, failsafes
  router.py                orchestration, concurrency
  local_model.py           optional local model (disabled by default)
  solvers/                 math, sentiment, ner, summarize, logic, code_tools
tests/test_all.py          11 offline tests (contract + routing invariants)
examples/tasks.json        one task per category + extras
docs/                      testing strategy, pitch
Dockerfile · requirements.txt
```
