from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compare_runs import compare_pair

from toolrl.io import load_tasks, read_jsonl
from toolrl.position_analysis import analyze_position_runs
from toolrl.replication import summarize_seed_replications


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate position-balanced SFT seed runs")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="LABEL=JSONL")
    parser.add_argument("--manifest", action="append", required=True, metavar="LABEL=JSON")
    parser.add_argument("--required-replication", action="append", required=True)
    parser.add_argument("--position-2-threshold", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = _named_paths(args.candidate)
    manifests = _named_paths(args.manifest)
    if set(candidates) != set(manifests):
        raise ValueError("candidate and manifest labels must match")

    tasks = load_tasks(args.tasks)
    baseline_rows = read_jsonl(args.baseline)
    reports = {}
    paired_comparisons = {}
    for label, candidate_path in candidates.items():
        paired_comparisons[label] = compare_pair(args.baseline, candidate_path)
        reports[label] = {
            "training_manifest": json.loads(manifests[label].read_text(encoding="utf-8")),
            "position_analysis": analyze_position_runs(
                tasks,
                baseline_rows,
                read_jsonl(candidate_path),
            ),
        }
    result = {
        "paired_comparisons": paired_comparisons,
        **summarize_seed_replications(
            reports,
            required_replications=args.required_replication,
            position_2_threshold=args.position_2_threshold,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def _named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"expected LABEL=PATH, got: {value}")
        if label in result:
            raise ValueError(f"duplicate label: {label}")
        result[label] = Path(raw_path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
