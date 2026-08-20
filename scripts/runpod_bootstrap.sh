#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v apt-get >/dev/null 2>&1; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 git
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --upgrade uv
fi

uv sync \
  --extra data \
  --extra train \
  --extra tracking \
  --extra hub \
  --extra eval \
  --extra export \
  --extra baselines \
  --extra demo

mkdir -p artifacts/cache artifacts/wandb
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add HF_TOKEN and WANDB_API_KEY before continuing."
else
  echo "Existing .env preserved."
fi

uv run turn-detector validate-config --config configs/runpod.yaml >/dev/null
echo "RunPod environment is ready. Next: edit .env, then run scripts/runpod_pipeline.sh."
