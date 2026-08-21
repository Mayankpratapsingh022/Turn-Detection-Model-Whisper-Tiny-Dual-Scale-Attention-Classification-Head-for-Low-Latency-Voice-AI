#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${DYNAMIC_CONFIG_PATH:-artifacts/configs/runpod.pinned.yaml}"
CHECKPOINT_DIR="${DYNAMIC_CHECKPOINT_DIR:-artifacts/checkpoints/e6_hard_negative/best}"
EXPORT_DIR="${DYNAMIC_EXPORT_DIR:-artifacts/export/e6_dynamic}"
EVALUATION_DIR="${DYNAMIC_EVALUATION_DIR:-artifacts/evaluation/e6_dynamic}"
RELEASE_DIR="${DYNAMIC_RELEASE_DIR:-artifacts/release/e6_dynamic}"
MODEL_PATH="$EXPORT_DIR/hinglish-turn.int8.onnx"
LOG_PATH="${DYNAMIC_LOG_PATH:-artifacts/dynamic-pipeline.log}"
START_AT="${DYNAMIC_START_AT:-calibrate}"
STOP_AFTER="${DYNAMIC_STOP_AFTER:-package}"
STAGES=(calibrate evaluate baselines package)

mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

# Preserve the CUDA-compatible Torch build installed on the pod. The remaining
# stages use ONNX Runtime on CPU and must not mutate the active environment.
export UV_NO_SYNC=1

stage_index() {
  local requested="$1"
  local index
  for index in "${!STAGES[@]}"; do
    if [[ "${STAGES[$index]}" == "$requested" ]]; then
      echo "$index"
      return 0
    fi
  done
  echo "Unknown dynamic stage: $requested" >&2
  exit 2
}

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing $description: $path" >&2
    exit 1
  fi
}

run_stage() {
  local name="$1"
  shift
  local index
  local started_at
  local finished_at
  local elapsed
  index="$(stage_index "$name")"
  if ((index < START_INDEX || index > STOP_INDEX)); then
    return 0
  fi
  started_at="$(date +%s)"
  echo "[$(date --iso-8601=seconds)] [dynamic:$name] START"
  if "$@"; then
    finished_at="$(date +%s)"
    elapsed="$((finished_at - started_at))"
    echo "[$(date --iso-8601=seconds)] [dynamic:$name] COMPLETE elapsed_seconds=$elapsed"
  else
    finished_at="$(date +%s)"
    elapsed="$((finished_at - started_at))"
    echo "[$(date --iso-8601=seconds)] [dynamic:$name] FAILED elapsed_seconds=$elapsed" >&2
    return 1
  fi
}

START_INDEX="$(stage_index "$START_AT")"
STOP_INDEX="$(stage_index "$STOP_AFTER")"
if ((START_INDEX > STOP_INDEX)); then
  echo "DYNAMIC_START_AT must not come after DYNAMIC_STOP_AFTER." >&2
  exit 2
fi

require_file "$CONFIG_PATH" "pinned RunPod config"
require_file "$CHECKPOINT_DIR/model.safetensors" "E6 best checkpoint"
require_file "$MODEL_PATH" "dynamic INT8 ONNX model"
require_file "$EXPORT_DIR/export_report.json" "dynamic export report"

"$REPO_ROOT/.venv/bin/python" -c "import json, pathlib; report=json.loads(pathlib.Path('$EXPORT_DIR/export_report.json').read_text()); assert report.get('quantization_method') == 'dynamic_weight_only', 'Expected dynamic_weight_only export'; assert report.get('meets_int8_probability_parity_target') is True, 'Dynamic INT8 parity target failed'; print('Dynamic export validated:', report['int8_path'], 'max_delta=', report['int8_max_probability_difference'])"

echo "[$(date --iso-8601=seconds)] [dynamic] START through=$STOP_AFTER log=$LOG_PATH"

run_stage calibrate uv run turn-detector calibrate \
  --model-path "$MODEL_PATH" \
  --config "$CONFIG_PATH" \
  --target-false-cutoff-rate 0.05

run_stage evaluate uv run turn-detector evaluate \
  --model-path "$MODEL_PATH" \
  --config "$CONFIG_PATH" \
  --output-dir "$EVALUATION_DIR" \
  --robustness

run_stage baselines uv run turn-detector compare-baselines \
  --model-path "$MODEL_PATH" \
  --config "$CONFIG_PATH" \
  --output-dir "$EVALUATION_DIR"

run_stage package uv run turn-detector package-model \
  --checkpoint "$CHECKPOINT_DIR" \
  --export-dir "$EXPORT_DIR" \
  --evaluation-dir "$EVALUATION_DIR" \
  --output "$RELEASE_DIR"

echo "[$(date --iso-8601=seconds)] [dynamic] COMPLETE through=$STOP_AFTER"
echo "Dynamic release: $RELEASE_DIR"
echo "No model was uploaded. Review the generated model card and reports before publishing."
