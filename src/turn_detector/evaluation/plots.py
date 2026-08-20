from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_pareto(results: list[dict[str, Any]], output_path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Plotting requires `uv sync --extra eval`") from exc
    if not results:
        raise ValueError("No policy results to plot")
    x = [100 * float(row["false_cutoff_rate"]) for row in results]
    y = [float(row["mean_endpoint_latency_ms"]) for row in results]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(x, y, marker="o", linewidth=1.5)
    axis.set_xlabel("False cutoff rate (%)")
    axis.set_ylabel("Mean endpoint latency (ms)")
    axis.set_title("Endpoint latency / interruption Pareto frontier")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)
