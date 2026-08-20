from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from turn_detector.config import AppConfig
from turn_detector.progress import log_event


def _resolved_revision(snapshot_path: str) -> str:
    path = Path(snapshot_path)
    return path.name if path.parent.name == "snapshots" else "unknown"


def cache_huggingface_assets(
    config: AppConfig,
    *,
    include_datasets: bool = True,
    include_model: bool = True,
    include_asr: bool = False,
    cache_dir: Path | None = None,
    manifest_path: Path = Path("artifacts/cache_manifest.json"),
) -> dict[str, Any]:
    """Prefetch immutable HF snapshots and record the revisions actually cached."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Asset caching requires `uv sync --extra hub`") from exc

    token = os.getenv("HF_TOKEN") or None
    selected_cache = str(cache_dir) if cache_dir is not None else None
    requests: list[tuple[str, str, str | None, str]] = []
    if include_datasets:
        requests.extend(
            [
                (config.data.train_repo, "dataset", config.data.train_revision, "train_dataset"),
                (config.data.test_repo, "dataset", config.data.test_revision, "test_dataset"),
            ]
        )
    if include_model:
        requests.append(
            (config.model.base_model, "model", config.model.base_model_revision, "base_model")
        )
    if include_asr:
        requests.append(("Systran/faster-whisper-large-v3", "model", None, "hinglish_asr"))
    if not requests:
        raise ValueError("At least one asset group must be selected")

    log_event(
        "cache",
        "START",
        assets=len(requests),
        cache_dir=selected_cache or os.getenv("HF_HOME") or "huggingface-default",
    )
    cached: list[dict[str, Any]] = []
    for repo_id, repo_type, revision, role in requests:
        log_event(
            f"cache:{role}",
            "DOWNLOAD_START",
            repo=repo_id,
            repo_type=repo_type,
            revision=revision or "main",
        )
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=selected_cache,
            token=token,
        )
        cached.append(
            {
                "role": role,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "requested_revision": revision or "main",
                "resolved_revision": _resolved_revision(snapshot_path),
                "snapshot_path": snapshot_path,
            }
        )
        log_event(
            f"cache:{role}",
            "DOWNLOAD_COMPLETE",
            resolved_revision=_resolved_revision(snapshot_path),
            snapshot=snapshot_path,
        )

    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "hf_home": os.getenv("HF_HOME"),
        "explicit_cache_dir": selected_cache,
        "assets": cached,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log_event("cache", "COMPLETE", assets=len(cached), manifest=manifest_path)
    return result


def pin_config_to_cached_revisions(
    config: AppConfig,
    cache_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write a config pinned to the exact train/test/model snapshots in a cache manifest."""
    payload = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    by_role = {asset["role"]: asset for asset in payload.get("assets", [])}
    expected = {
        "train_dataset": config.data.train_repo,
        "test_dataset": config.data.test_repo,
        "base_model": config.model.base_model,
    }
    for role, repo_id in expected.items():
        asset = by_role.get(role)
        if asset is None:
            raise ValueError(f"Cache manifest has no {role} snapshot")
        if asset.get("repo_id") != repo_id:
            raise ValueError(
                f"Cache manifest {role} repo {asset.get('repo_id')!r} does not match {repo_id!r}"
            )
        if asset.get("resolved_revision") in {None, "", "unknown"}:
            raise ValueError(f"Cache manifest has no resolved revision for {role}")

    pinned = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "train_revision": by_role["train_dataset"]["resolved_revision"],
                    "test_revision": by_role["test_dataset"]["resolved_revision"],
                }
            ),
            "model": config.model.model_copy(
                update={"base_model_revision": by_role["base_model"]["resolved_revision"]}
            ),
        }
    )
    pinned.to_yaml(output_path)
    result = {
        "output_path": str(output_path),
        "train_revision": pinned.data.train_revision,
        "test_revision": pinned.data.test_revision,
        "base_model_revision": pinned.model.base_model_revision,
    }
    log_event("cache:pin-config", "COMPLETE", **result)
    return result
