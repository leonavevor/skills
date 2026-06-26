"""Summarize benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize metrics across multiple runs."""
    if not runs:
        return {}

    metrics = {}
    sample = runs[0]

    for key, value in sample.items():
        if key == "name" or key == "id":
            continue
        if not isinstance(value, (int, float)):
            continue

        values = [r.get(key, 0) for r in runs if isinstance(r.get(key), (int, float))]
        if values:
            metrics[f"{key}_mean"] = sum(values) / len(values)
            metrics[f"{key}_min"] = min(values)
            metrics[f"{key}_max"] = max(values)

    return metrics


def run(input_path: Path, output_path: Path) -> None:
    """Load benchmark results and generate summary."""
    data = json.loads(input_path.read_text())

    summary = {}
    if "runs" in data:
        summary = summarize_metrics(data["runs"])

    result = {
        "total_runs": len(data.get("runs", [])),
        "summary": summary,
    }

    output_path.write_text(json.dumps(result, indent=2))
    print(f"Results summary written to {output_path}")

