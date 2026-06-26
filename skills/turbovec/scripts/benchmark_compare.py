"""Benchmark comparison utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple


class BenchmarkRun(NamedTuple):
    """A single benchmark run result."""

    name: str
    metrics: dict[str, Any]


def parse_runs(data: dict[str, Any]) -> list[BenchmarkRun]:
    """Parse benchmark runs from JSON."""
    runs = []
    for run in data.get("runs", []):
        runs.append(BenchmarkRun(name=run["name"], metrics=run))
    return runs


def generate_markdown(runs: list[BenchmarkRun]) -> str:
    """Generate a markdown comparison report."""
    lines = ["# TurboVec Benchmark Comparison\n"]

    if not runs:
        return "\n".join(lines) + "No runs found."

    # Header
    lines.append("| Metric | " + " | ".join(r.name for r in runs) + " |")
    lines.append("|--------|" + "|".join("---" for _ in runs) + "|")

    # Metrics
    all_keys = set()
    for run in runs:
        all_keys.update(run.metrics.keys())

    for key in sorted(all_keys):
        if key == "name":
            continue
        row = [f"__{key}__"]
        for run in runs:
            val = run.metrics.get(key, "N/A")
            if isinstance(val, float):
                row.append(f"{val:.2f}")
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def run(input_path: Path, output_path: Path) -> None:
    """Load benchmark JSON and generate report."""
    data = json.loads(input_path.read_text())
    runs = parse_runs(data)
    report = generate_markdown(runs)
    output_path.write_text(report)
    print(f"Benchmark report written to {output_path}")

