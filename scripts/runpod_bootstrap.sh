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

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! .venv/bin/python -c "import torch; assert torch.cuda.is_available(), f'CUDA unavailable: torch={torch.__version__}, torch_cuda={torch.version.cuda}'; print(f'CUDA preflight passed: torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}')"; then
    echo "RunPod CUDA preflight failed. Do not start training on CPU." >&2
    echo "Install a PyTorch build compatible with the host driver before continuing." >&2
    exit 1
  fi
fi

mkdir -p artifacts/cache artifacts/wandb
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add HF_TOKEN and WANDB_API_KEY before continuing."
else
  echo "Existing .env preserved."
fi

uv run turn-detector validate-config --config configs/runpod.yaml >/dev/null
echo "RunPod environment is ready. Next: edit .env, then run scripts/runpod_pipeline.sh."
