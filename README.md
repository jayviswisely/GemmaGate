# GemmaGate

Validation-gated token router for the AMD Developer Hackathon Track 1:
Hybrid Token-Efficient Routing Agent.

GemmaGate is a local-first AI agent that solves tasks inside the container when
the answer can be verified for free, then calls Fireworks AI only when local
validation says escalation is needed. The goal is to preserve accuracy while
spending as few Fireworks tokens as possible.

## Submission

Public Docker image:

```text
jayviswisely/gemmagate:latest
```

Docker Hub:

```text
https://hub.docker.com/r/jayviswisely/gemmagate
```

GitHub repository:

```text
https://github.com/jayviswisely/GemmaGate
```

## What It Handles

GemmaGate routes across all eight Track 1 capability categories:

| Category | Local strategy | Remote strategy |
| --- | --- | --- |
| Factual knowledge | Optional local LLM consensus; short-answer factual residue can be batched | Cheap model first, stronger only if needed |
| Mathematical reasoning | Deterministic math solver, numeric validation, judge-friendly answer polish | Escalate only on unsupported patterns |
| Sentiment classification | Lexicon, negation, contrast, sarcasm guards, optional local LLM self-check | Batch uncertain label-only cases |
| Text summarization | Extractive summaries plus optional local LLM verification | Cheap or mid model for open-ended summaries |
| Named entity recognition | Regex and gazetteer extraction with coverage guard | JSON-mode remote call if local coverage is uncertain |
| Code debugging | Mechanical repairs and executable validation | Mid or strong model for harder fixes |
| Logical reasoning | Brute-force constraint solver where provable; optional local LLM consensus | Stronger model only when constraints are not parseable |
| Code generation | Safe templates plus syntax/self tests; optional local LLM examples gate | Mid or strong model for unknown specs |

## Why It Is Token Efficient

Most submissions send every task directly to a large model. GemmaGate treats the
problem as a verification task:

1. Classify the prompt locally.
2. Try the cheapest local solver for that task type.
3. Validate the answer with deterministic checks.
4. Polish zero-token local answers when that helps an LLM judge parse them.
5. If a bundled GGUF model is present, sample it locally and accept only after
   validation, agreement, and a tiny local YES/NO verification pass.
6. Keep math and hard NER on deterministic proof or Fireworks by default,
   because plausible unchecked local guesses are risky for the accuracy gate.
7. Batch eligible sentiment and short factual residue to amortize prompt cost.
8. Prefer `kimi-k2p7` by substring when it appears in `ALLOWED_MODELS`, based
   on observed Track 1 bakeoff behavior.
9. Disable hidden reasoning with `reasoning_effort="none"` on Fireworks calls
   when the endpoint accepts that parameter.
10. Otherwise call the cheapest allowed Fireworks model.
11. Escalate to larger models only if validation fails.
12. On remote validation failure, retry with the original task context plus the
    validation error instead of a context-free repair fragment.

The default image is Python standard-library only, starts quickly, and avoids
bundling a large local model. This keeps the container small and safe for the
hackathon grading environment. `Dockerfile.local` is the optional leaderboard
profile: it bakes a Qwen2.5-3B GGUF into the image and enables the local
consensus gate.

## Current Local Checks

Current offline validation:

```text
Unit tests: 56/56
Dry benchmark: 21/23 locally scored tasks
Estimated dry-run remote tokens: 144
Solved fully locally in dry run: 21/23
```

Dry-run remote answers are simulated, so the dry benchmark is useful for
checking routing and local solvers, not final hidden accuracy. Use the real
benchmark before pushing a submission image.

## Input And Output Contract

The container reads:

```text
/input/tasks.json
```

Expected input format:

```json
[
  {
    "task_id": "t1",
    "prompt": "Summarize the following text in one sentence: ..."
  }
]
```

The container writes:

```text
/output/results.json
```

Expected output format:

```json
[
  {
    "task_id": "t1",
    "answer": "..."
  }
]
```

GemmaGate writes a placeholder `results.json` before solving begins, then
replaces it atomically with final results. This protects against missing output
files if a task fails or the process is interrupted.

## Required Environment Variables

The hackathon harness provides these at runtime:

```text
FIREWORKS_API_KEY
FIREWORKS_BASE_URL
ALLOWED_MODELS
```

Optional routing and remote-call controls:

```text
GEMMAGATE_MODEL_PIN=kimi-k2p7   # default substring preference if present
GEMMAGATE_MODEL_PIN=            # disable model pinning
GEMMAGATE_REASONING_OFF=1       # default; send reasoning_effort="none"
GEMMAGATE_REASONING_OFF=0       # disable reasoning parameter
GEMMAGATE_LOCAL_GGUF=/models/model.gguf
GEMMAGATE_LOCAL_SAMPLES=3
GEMMAGATE_LOCAL_MIN_AGREE=2
GEMMAGATE_LOCAL_VERIFY=1
GEMMAGATE_LOCAL_FULL=1          # optional: allow local model for math/NER too
```

Do not hardcode secrets or model IDs in the repository. For local development,
use a private `.env.local` file and keep it out of Git.

Example local `.env.local`:

```env
FIREWORKS_API_KEY=your_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=model-a,model-b,model-c
```

## Run With Docker

Build the local image:

```powershell
docker build -t gemmagate .
```

Build the bundled-local-model image:

```powershell
scripts\get_model.ps1
docker build -f Dockerfile.local -t gemmagate:local .
```

Run it like the judging harness:

```powershell
New-Item -ItemType Directory -Force out

docker run --rm `
  --env-file .env.local `
  -v "${PWD}\examples:/input:ro" `
  -v "${PWD}\out:/output" `
  gemmagate
```

Use `gemmagate:local` in the command above when testing the bundled GGUF
profile.

Inspect the result:

```powershell
Get-Content out\results.json
```

Run the published image:

```powershell
docker run --rm `
  --env-file .env.local `
  -v "${PWD}\examples:/input:ro" `
  -v "${PWD}\out:/output" `
  jayviswisely/gemmagate:latest
```

## Build And Push Submission Image

The judging VM runs `linux/amd64`, so the submitted image must include a
`linux/amd64` manifest.

```powershell
docker login
docker buildx build --platform linux/amd64 -t jayviswisely/gemmagate:latest --push .
```

For the bundled-local-model submission profile:

```powershell
scripts\get_model.ps1
docker buildx build --platform linux/amd64 -f Dockerfile.local -t jayviswisely/gemmagate:latest --push .
```

Verify that the public image can be pulled:

```powershell
docker pull jayviswisely/gemmagate:latest
```

## Local Testing

Run the offline unit suite:

```powershell
python tests\test_all.py
```

Run the dry benchmark:

```powershell
python eval\run_benchmark.py
```

Run the real Fireworks benchmark. This spends real tokens:

```powershell
python eval\run_benchmark.py --real --skip-baseline
```

Use `--skip-baseline` when you only want to test GemmaGate and avoid extra
Fireworks calls from the naive baseline.

## Architecture

```text
main.py
  - reads /input/tasks.json
  - writes placeholder /output/results.json immediately
  - runs the router under a wall-clock deadline
  - writes final results atomically

gemmagate/io_layer.py
  - validates and salvages task input
  - guarantees valid JSON output

gemmagate/classifier.py
  - classifies prompts into the eight task categories
  - extracts task structure such as payloads, labels, JSON requirements,
    sentence limits, word limits, and code language

gemmagate/router.py
  - plans model tiers from ALLOWED_MODELS
  - runs tasks concurrently
  - batches eligible sentiment and short factual residue
  - isolates per-task failures

gemmagate/escalation.py
  - walks the local -> cheap -> mid -> strong ladder
  - accepts local LLM answers only after validation, consensus, and self-checks
  - returns a non-empty failsafe answer if all else fails

gemmagate/local_model.py
  - loads optional GGUF or Transformers models from environment variables
  - serializes local generation calls for thread-safe CPU inference

gemmagate/validator.py
  - checks answers for format, constraints, and task-specific correctness
  - repairs simple formatting issues for free

gemmagate/present.py
  - polishes zero-token local answers only when the prompt does not require
    a bare label, word, number, or JSON response

gemmagate/batcher.py
  - batches eligible sentiment and short factual tasks to reduce repeated
    instruction overhead

gemmagate/remote.py
  - calls Fireworks through FIREWORKS_BASE_URL
  - records token usage
  - supports dry-run testing without network access
```

## Repository Layout

```text
main.py
Dockerfile
Dockerfile.local
requirements.txt
gemmagate/
  batcher.py
  classifier.py
  escalation.py
  io_layer.py
  local_model.py
  model_select.py
  present.py
  prompts.py
  remote.py
  router.py
  schemas.py
  validator.py
  solvers/
tests/
eval/
examples/
docs/
```

## Notes

- The default Dockerfile is the fast, small submission path.
- `Dockerfile.local` is the optional bundled-local-model profile. It is larger
  but can reduce Fireworks usage because in-container inference scores as zero
  remote tokens.
- `.env.local`, root `results.json`, generated outputs, and Python cache files
  should never be committed.
- The agent reads model IDs from `ALLOWED_MODELS` at runtime and does not
  hardcode launch-day model names.
- `GEMMAGATE_MODEL_PIN` is a substring preference only; if no allowed model
  matches it, GemmaGate falls back to normal tier planning.
- Do not set `GEMMAGATE_ACCURACY_FIRST` for the current submission path; that
  experiment was removed from the Docker image after leaderboard testing.

## License

MIT License. See `LICENSE`.
