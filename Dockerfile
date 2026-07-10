# GemmaGate — competition submission image
#
# TWO BUILD PROFILES:
#
# (A) Minimal (default, ~150MB): stdlib-only. Heuristics + rule solvers local,
#     Fireworks for the rest. Zero cold-start cost, zero risk.
#
# (B) Draft-enabled (~2GB, still far under the 10GB cap): bakes a small GGUF
#     so the draft-conditional rung activates — local model drafts factual/
#     summary/logic answers, Fireworks verifies with "OK" (1-2 output tokens)
#     or corrects in the same call. Uncomment the three PROFILE B blocks and
#     place the GGUF at models/ before building, e.g.:
#       huggingface-cli download bartowski/gemma-2-2b-it-GGUF \
#         gemma-2-2b-it-Q4_K_M.gguf --local-dir models/
# REQUIRED by the judging harness: build for linux/amd64
#   docker buildx build --platform linux/amd64 -t <tag> .
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app
# Profile A runtime is Python STDLIB ONLY — no pip step, nothing to fail,
# no supply-chain surface. requirements.txt documents optional extras.
COPY requirements.txt .

# --- PROFILE B (1/3): build deps + llama-cpp
# RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ cmake \
#     && pip install --no-cache-dir llama-cpp-python==0.2.90 \
#     && apt-get purge -y gcc g++ cmake && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY gemmagate/ ./gemmagate/
COPY main.py .

# --- PROFILE B (2/3): bake the weights
# COPY models/gemma-2-2b-it-Q4_K_M.gguf /models/model.gguf

# --- PROFILE B (3/3): activate
# ENV GEMMAGATE_LOCAL_GGUF=/models/model.gguf

# The harness injects FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS.
ENTRYPOINT ["python", "main.py"]
