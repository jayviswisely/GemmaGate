# GemmaGate

Validation-gated token router for the AMD Developer Hackathon Track 1:
Hybrid Token-Efficient Routing Agent.

GemmaGate is a local-first AI agent that solves tasks inside the container when
the answer can be verified for free, then calls Fireworks AI only when local
validation says escalation is needed. The goal is to preserve accuracy while
spending as few Fireworks tokens as possible.

For the current submission, real runs default to `GEMMAGATE_QUALIFIER_MODE=1`.
That mode keeps provable math and logic local, but sends hidden-risky categories
to the strongest available Fireworks tier first. This spends more tokens than
the dry benchmark on purpose because the leaderboard accuracy gate must be
passed before token reduction matters.

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
| Factual knowledge | Never guessed locally; short-answer factual residue can be batched | Cheap model first, stronger only if needed |
| Mathematical reasoning | Deterministic math solver, numeric validation, judge-friendly answer polish | Escalate only on unsupported patterns |
| Sentiment classification | Lexicon, negation, contrast, sarcasm guards, grounded justifications | Batch uncertain label-only cases |
| Text summarization | Extractive summaries for checkable constraints | Cheap or mid model for open-ended summaries |
| Named entity recognition | Regex and gazetteer extraction with coverage guard | JSON-mode remote call if local coverage is uncertain |
| Code debugging | Mechanical repairs and executable validation | Mid or strong model for harder fixes |
| Logical reasoning | Brute-force constraint solver where provable | Stronger model only when constraints are not parseable |
| Code generation | Safe templates plus syntax/self tests | Mid or strong model for unknown specs |

## Why It Is Token Efficient

Most submissions send every task directly to a large model. GemmaGate treats the
problem as a verification task:

1. Classify the prompt locally.
2. Try the cheapest local solver for that task type.
3. Validate the answer with deterministic checks.
4. Polish zero-token local answers when that helps an LLM judge parse them.
5. Accept the local answer only when validation passes.
6. Batch eligible sentiment and short factual residue to amortize prompt cost.
7. In dry-run/token mode, otherwise call the cheapest allowed Fireworks model.
8. Escalate to larger models only if validation fails.

In qualifier mode, GemmaGate changes that order for hidden-risky categories:
sentiment, NER, summarization, factual knowledge, code debugging, and code
generation go to the strongest tier first. This is less token-efficient, but it
is the safer path when the target is an 84-85% accuracy gate.

The default image is Python standard-library only, starts quickly, and avoids
bundling a large local model. This keeps the container small and safe for the
hackathon grading environment.

## Current Local Checks

Current offline validation:

```text
Unit tests: 54/54
Dry benchmark: 21/23 locally scored tasks
Estimated dry-run remote tokens: 115
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

Optional routing override:

```text
GEMMAGATE_QUALIFIER_MODE=1  # accuracy-first, default for real submissions
GEMMAGATE_QUALIFIER_MODE=0  # local-first token mode
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

Run it like the judging harness:

```powershell
New-Item -ItemType Directory -Force out

docker run --rm `
  --env-file .env.local `
  -v "${PWD}\examples:/input:ro" `
  -v "${PWD}\out:/output" `
  gemmagate
```

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

Real runs default to qualifier mode. To compare the older local-first token
mode, run:

```powershell
$env:GEMMAGATE_QUALIFIER_MODE = "0"
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
  - accepts answers only after validation
  - returns a non-empty failsafe answer if all else fails

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

- The default Dockerfile is the recommended submission path.
- `Dockerfile.local` is an optional local-model profile and is not required for
  the default submission.
- `.env.local`, root `results.json`, generated outputs, and Python cache files
  should never be committed.
- The agent reads model IDs from `ALLOWED_MODELS` at runtime and does not
  hardcode launch-day model names.
- Do not set `GEMMAGATE_ACCURACY_FIRST`; the current accuracy-focused switch is
  `GEMMAGATE_QUALIFIER_MODE`.

## License

MIT License. See `LICENSE`.
