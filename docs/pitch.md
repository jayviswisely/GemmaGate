# GemmaGate — Final Pitch Explanation

## The one-sentence idea
Every task falls down a cost-ordered ladder — deterministic Python, then rule solvers, then the cheapest allowed Fireworks model, then stronger ones — and a **free local validator** is the gatekeeper that decides whether the climb stops; we only pay tokens when a free check proves the free answer wasn't good enough.

## Why this framing wins the scoring function
The score charges only remote tokens and gates on accuracy first. Most teams will read that as a *model selection* problem: predict which model each task needs. Prediction is error-prone in both directions — over-routing wastes tokens, under-routing fails the gate. We reframed it as a *verification* problem: for most benchmark formats (numbers, labels, JSON entity sets, length-capped summaries, ordering puzzles, parseable code), checking an answer is deterministic and free even when producing one isn't. So we produce cheaply and verify rigorously, escalating only on proven failure.

## Three things judges should notice

**1. The local rungs aren't "a small model" — they're provably-correct tools with refusal discipline.**
Math is re-derived through an AST-whitelisted evaluator; logic puzzles are brute-forced over every permutation, so a returned answer is a theorem, not a guess. Every local solver is built to say "not my task" on ambiguity: the sentiment lexicon abstains on mixed signals, the NER extractor abstains if any capitalized span went unlabeled (coverage guard), the summarizer only fires on checkable constraints. That refusal discipline is what lets 60–70% local coverage coexist with an accuracy gate.

**2. Validation is also a weapon against remote mistakes.**
When the cheap model answers a math or logic task, the validator cross-checks it against the local derivation. On conflict, the repair isn't another paid round-trip of the full task — the validator *already knows the right answer* and substitutes it. When repairs do need the model, they send only the failed output plus the violated constraint (~30–120 tokens), never the original prompt again.

**3. The harness contract is engineered, not assumed.**
A placeholder `results.json` is written before solving begins; writes are atomic; task_ids are regex-salvaged from even syntactically broken input; every task runs inside its own exception boundary; a deadline manager degrades gracefully to local best-effort answers in the final seconds. The agent cannot score zero on a technicality.

## Honest trade-offs we chose
- **Math gets brief working.** Pure answer-only prompting on small models tanks arithmetic accuracy, and accuracy ranks first — so math prompts allow minimal reasoning ending in `ANSWER: <value>` (capped at 320 output tokens) and the validator extracts the value. A few tokens bought accuracy where it matters.
- **Facts are never guessed locally.** No lexicon can verify a factual claim, so factual QA always uses the cheap remote tier with a concise-answer cap. Spending ~60 tokens beats gambling the accuracy gate.
- **Code skips the cheap tier.** A cheap model that predictably fails at code still bills its tokens before escalation — expected-value routing says start at mid.

## Robustness to launch-day unknowns
No model IDs anywhere in code. Tiers are computed at startup from `ALLOWED_MODELS` name hints (parameter counts, MoE patterns, size keywords; unknown names default to mid — never accidentally cheap), overridable via env substrings in seconds. Base URL and key come from env only. The whole agent runs offline in dry-run mode, so the routing logic was fully tested before a single real token was spent.
