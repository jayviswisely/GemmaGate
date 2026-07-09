#!/usr/bin/env bash
set -e
mkdir -p models
curl -L -o models/model.gguf \
  "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
echo "Done -> models/model.gguf"
