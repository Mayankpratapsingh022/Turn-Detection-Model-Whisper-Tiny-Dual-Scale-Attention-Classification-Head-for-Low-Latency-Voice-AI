#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f .env ]]; then
  echo "Missing .env. Run scripts/runpod_bootstrap.sh and add the required keys." >&2
  exit 1
fi

START_AT="${PIPELINE_START_AT:-cache}"
STOP_AFTER="${PIPELINE_STOP_AFTER:-package}"
STAGES=(cache prepare train_e5 mine train_e6 export calibrate evaluate baselines package)

stage_index() {
  local requested="$1"
  local index
  for index in "${!STAGES[@]}"; do
    if [[ "${STAGES[$index]}" == "$requested" ]]; then
      echo "$index"
      return 0
    fi
  done
  echo "Unknown stage: $requested" >&2
  exit 2
}

START_INDEX="$(stage_index "$START_AT")"
STOP_INDEX="$(stage_index "$STOP_AFTER")"
if (( START_INDEX > STOP_INDEX )); then
  echo "PIPELINE_START_AT must not come after PIPELINE_STOP_AFTER." >&2
  exit 2
fi

run_stage() {
  local name="$1"
  shift
  local index
  index="$(stage_index "$name")"
  if (( index < START_INDEX || index > STOP_INDEX )); then
    return 0
  fi
  local started_at
  local finished_at
  local elapsed
  started_at="$(date +%s)"
  echo "[$(date --iso-8601=seconds)] [pipeline:$name] START"
  if "$@"; then
    finished_at="$(date +%s)"
    elapsed="$((finished_at - started_at))"
    echo "[$(date --iso-8601=seconds)] [pipeline:$name] COMPLETE elapsed_seconds=$elapsed"
  else
    finished_at="$(date +%s)"
    elapsed="$((finished_at - started_at))"
    echo "[$(date --iso-8601=seconds)] [pipeline:$name] FAILED elapsed_seconds=$elapsed" >&2
    return 1
  fi
}

cache_and_pin() {
  uv run turn-detector cache-assets --config configs/runpod.yaml
  uv run turn-detector pin-config --config configs/runpod.yaml \
    --output artifacts/configs/runpod.pinned.yaml
  uv run turn-detector pin-config --config configs/experiments/e5_causal_filler.yaml \
    --output artifacts/configs/e5_causal_filler.pinned.yaml
  uv run turn-detector pin-config --config configs/experiments/e6_hard_negative.yaml \
    --output artifacts/configs/e6_hard_negative.pinned.yaml
}

run_stage cache cache_and_pin
run_stage prepare uv run turn-detector data prepare \
  --config artifacts/configs/runpod.pinned.yaml
run_stage train_e5 uv run turn-detector train \
  --config artifacts/configs/e5_causal_filler.pinned.yaml
run_stage mine uv run turn-detector mine-hard-negatives \
  --model-path artifacts/checkpoints/e5_causal_filler/best \
  --manifest artifacts/data/train.jsonl \
  --output artifacts/data/hard_negatives.jsonl \
  --config artifacts/configs/runpod.pinned.yaml
run_stage train_e6 uv run turn-detector train \
  --config artifacts/configs/e6_hard_negative.pinned.yaml
run_stage export uv run turn-detector export \
  --checkpoint artifacts/checkpoints/e6_hard_negative/best \
  --output artifacts/export/e6/hinglish-turn.onnx \
  --config artifacts/configs/runpod.pinned.yaml --quantize --static
run_stage calibrate uv run turn-detector calibrate \
  --model-path artifacts/export/e6/hinglish-turn.int8.onnx \
  --config artifacts/configs/runpod.pinned.yaml --target-false-cutoff-rate 0.05
run_stage evaluate uv run turn-detector evaluate \
  --model-path artifacts/export/e6/hinglish-turn.int8.onnx \
  --config artifacts/configs/runpod.pinned.yaml \
  --output-dir artifacts/evaluation/e6 --robustness
run_stage baselines uv run turn-detector compare-baselines \
  --model-path artifacts/export/e6/hinglish-turn.int8.onnx \
  --config artifacts/configs/runpod.pinned.yaml \
  --output-dir artifacts/evaluation/e6
run_stage package uv run turn-detector package-model \
  --checkpoint artifacts/checkpoints/e6_hard_negative/best \
  --export-dir artifacts/export/e6 \
  --evaluation-dir artifacts/evaluation/e6 \
  --output artifacts/release/e6

echo "[$(date --iso-8601=seconds)] [pipeline] COMPLETE through=$STOP_AFTER"
echo "No model was uploaded. Review the reports/model card, then run the explicit push-model command."
