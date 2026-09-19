#!/usr/bin/env python3
"""Compare paired two-GPU Riemannian benchmarks against acceptance gates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

VALUE_PATTERN = re.compile(r"^(?P<key>[A-Za-z0-9_@.-]+)=(?P<value>[-+0-9.eE]+)$")
METRIC_PATTERN = re.compile(
    r"(?:^| - )(?P<key>mrr|hits@[0-9]+):\s*(?P<value>[-+0-9.eE]+)$"
)


def parse_summary(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = VALUE_PATTERN.search(line) or METRIC_PATTERN.search(line)
        if match is not None:
            values[match.group("key")] = float(match.group("value"))
    return values


def peak_memory(values: dict[str, float]) -> float:
    peaks = [
        value
        for key, value in values.items()
        if key.startswith("gpu_") and key.endswith("_peak_memory_mib")
    ]
    if not peaks:
        raise ValueError("No GPU peak-memory values were found")
    return max(peaks)


def require_value(values: dict[str, float], key: str, source: Path) -> float:
    if key not in values:
        raise ValueError(f"{source} does not contain {key}")
    return values[key]


def build_report(
    baseline_train: dict[str, float],
    optimized_train: dict[str, float],
    baseline_eval: dict[str, float],
    optimized_eval: dict[str, float],
    *,
    memory_ratio: float,
    metric_tolerance: float,
) -> dict[str, object]:
    baseline_elapsed = baseline_train["elapsed_seconds"]
    optimized_elapsed = optimized_train["elapsed_seconds"]
    baseline_peak = peak_memory(baseline_train)
    optimized_peak = peak_memory(optimized_train)

    metric_deltas = {
        metric: optimized_eval[metric] - baseline_eval[metric]
        for metric in ("mrr", "hits@1", "hits@3", "hits@10")
    }
    checks = {
        "memory_at_most_configured_ratio": (
            optimized_peak <= baseline_peak * memory_ratio
        ),
        "elapsed_time_not_slower": optimized_elapsed <= baseline_elapsed,
        "metrics_within_tolerance": all(
            abs(delta) <= metric_tolerance for delta in metric_deltas.values()
        ),
    }
    return {
        "baseline_elapsed_seconds": baseline_elapsed,
        "optimized_elapsed_seconds": optimized_elapsed,
        "speedup_ratio": baseline_elapsed / optimized_elapsed,
        "baseline_peak_memory_mib": baseline_peak,
        "optimized_peak_memory_mib": optimized_peak,
        "memory_reduction_ratio": 1.0 - optimized_peak / baseline_peak,
        "metric_deltas": metric_deltas,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-train", type=Path, required=True)
    parser.add_argument("--optimized-train", type=Path, required=True)
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--optimized-eval", type=Path, required=True)
    parser.add_argument("--memory-ratio", type=float, default=0.75)
    parser.add_argument("--metric-tolerance", type=float, default=1e-4)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    baseline_train = parse_summary(args.baseline_train)
    optimized_train = parse_summary(args.optimized_train)
    baseline_eval = parse_summary(args.baseline_eval)
    optimized_eval = parse_summary(args.optimized_eval)

    require_value(baseline_train, "elapsed_seconds", args.baseline_train)
    require_value(optimized_train, "elapsed_seconds", args.optimized_train)
    for metric in ("mrr", "hits@1", "hits@3", "hits@10"):
        require_value(baseline_eval, metric, args.baseline_eval)
        require_value(optimized_eval, metric, args.optimized_eval)

    report = build_report(
        baseline_train,
        optimized_train,
        baseline_eval,
        optimized_eval,
        memory_ratio=args.memory_ratio,
        metric_tolerance=args.metric_tolerance,
    )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
