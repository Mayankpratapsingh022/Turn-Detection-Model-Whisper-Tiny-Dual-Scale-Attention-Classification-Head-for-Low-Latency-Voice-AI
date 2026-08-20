# RunPod training runbook

This runbook keeps every large artifact on persistent storage and every credential outside source
control. None of the repository commands creates a RunPod pod, logs into an account, or publishes a
model automatically.

## Pod and storage

Recommended starting point:

- GPU: one A100 40 GB; L40S/A40/RTX 4090 are acceptable alternatives.
- CPU: 12–16 vCPUs.
- RAM: 48–64 GB.
- Persistent disk: 200 GB recommended; 150 GB is the practical minimum.
- Container disk: 20–30 GB is enough when `/workspace`, `HF_HOME`, and artifacts are persistent.
- Image: a current CUDA 12.x PyTorch image with Python 3.11 or 3.12.

The default physical batch is 32 and effective batch is 256 through gradient accumulation. On an
out-of-memory error, change only `physical_batch_size` in `configs/runpod.yaml` to 16 or 8 and leave
the effective batch at 256.

Actual wall time depends mostly on pod download bandwidth, total decoded audio duration, and GPU
type. Training starts directly from the prepared dataset metadata; a full-corpus ASR transcription
pass is not required. A repeat run with warm Hugging Face and prepared-audio caches is usually much
faster. The two training stages are the dominant GPU work. Do not rent several GPUs: the trainer is
deliberately single-GPU, and one fast GPU is the cost-efficient setup.

## Bootstrap and credentials

From the cloned repository on `/workspace`:

```bash
bash scripts/runpod_bootstrap.sh
nano .env
```

Required for the standard online workflow:

- `HF_TOKEN`: a Hugging Face user access token. Read access is enough for public downloads; write
  access is required only for the later `push-model` command.
- `WANDB_API_KEY`: a W&B API key.
- `HF_MODEL_REPO`: the intended `owner/model` destination.
- `WANDB_ENTITY` and `WANDB_PROJECT`: the W&B destination.

Keep `HF_HOME=/workspace/cache/huggingface` and `WANDB_DIR=/workspace/artifacts/wandb` on the
persistent volume. Never paste tokens into YAML, shell scripts, notebooks, reports, or command-line
arguments.

## Download and verify inputs

```bash
uv run turn-detector cache-assets --config configs/runpod.yaml
uv run turn-detector validate-config --config configs/runpod.yaml
```

Inspect `artifacts/cache_manifest.json`. Before the final experiment, copy the recorded resolved
commit hashes into a reviewed config, or generate it directly:

```bash
uv run turn-detector pin-config \
  --config configs/runpod.yaml \
  --cache-manifest artifacts/cache_manifest.json \
  --output artifacts/configs/runpod.pinned.yaml
```

`scripts/runpod_pipeline.sh` does this automatically for the RunPod, E5, and E6 configs before any
data preparation or training. This converts the convenient download against `main` into a
reproducible pinned run.

To cache only a subset:

```bash
uv run turn-detector cache-assets --no-datasets --model
uv run turn-detector cache-assets --datasets --no-model
```

`--asr` is an explicit opt-in for exploratory transcript analysis and is not used by training.

## Run and monitor

```bash
bash scripts/runpod_pipeline.sh 2>&1 | tee artifacts/runpod-pipeline.log
```

Useful checks in another terminal:

```bash
nvidia-smi
du -sh artifacts /workspace/cache/huggingface
tail -f artifacts/runpod-pipeline.log
```

W&B receives training and validation metrics at the configured step intervals and the selected best
checkpoint. Local `training_report.json` remains the source-of-truth fallback. If W&B is unavailable,
set `WANDB_MODE=offline`; the training job still writes an offline W&B run under `WANDB_DIR`.

The terminal and `artifacts/runpod-pipeline.log` also contain live progress for every expensive
operation. Typical lines look like:

```text
prepare:train:  31%|...| 14820/48000 [accepted=12104 records=17882 excluded=2380 rejected=336]
epoch 2/4: 64%|...| 820/1280 [step=742 loss=0.1842 main=0.1661 filler=0.1205]
[2026-08-20T18:31:00+00:00] [pipeline:train_e5] COMPLETE elapsed_seconds=5142
```

Hugging Face streaming metadata supplies the preparation denominator when the dataset publishes its
split count. All manifest-backed stages have exact totals. Progress events never include API keys or
transcripts.

The pipeline can be bounded or restarted at a stage:

```bash
PIPELINE_START_AT=train_e5 PIPELINE_STOP_AFTER=train_e5 bash scripts/runpod_pipeline.sh
PIPELINE_START_AT=mine bash scripts/runpod_pipeline.sh
```

Preparation is cache aware. The model trainer does not yet reconstruct an interrupted
optimizer/data-loader position, so a restarted `train_e5` or `train_e6` stage is a clean new
training run. Checkpoints are still retained for audit and inference.

## Promotion gate

Do not choose a model from test accuracy alone. Promote E6 only after reviewing:

- causal false-cutoff rate at 300 and 600 ms;
- endpoint latency under 5% and 10% false-cutoff budgets;
- Hindi, Indian English, high-confidence Hinglish, filler, long-pause, and sequence slices;
- calibration error and reliability plot;
- noise, clipping, gain, resampling, and silence robustness;
- bootstrap confidence intervals;
- fixed-timeout and Smart Turn v3.2 comparisons;
- FP32/INT8 parity, CPU latency, and the 10 MB deployment-size target.

The full evidence lands in `artifacts/evaluation/e6`, `artifacts/export/e6`, and the staged release
folder.

## Package, review, and upload

```bash
uv run turn-detector package-model \
  --checkpoint artifacts/checkpoints/e6_hard_negative/best \
  --export-dir artifacts/export/e6 \
  --evaluation-dir artifacts/evaluation/e6 \
  --output artifacts/release/e6
```

Edit the staged model card with actual dataset counts, revisions, metrics, limitations, and hardware.
Verify every SHA-256 entry in `release_manifest.json` corresponds to an intended file. The package
excludes `.env`, source audio, and optimizer state.

Only after confirming the upstream dataset terms:

```bash
uv run turn-detector push-model \
  --folder artifacts/release/e6 \
  --repo-id "$HF_MODEL_REPO" \
  --private \
  --acknowledge-source-license-review
```

The upload API sends only the allow-listed files in `release_manifest.json`. Change `--private` to
`--public` only as a deliberate final action. Training never calls the upload function.
