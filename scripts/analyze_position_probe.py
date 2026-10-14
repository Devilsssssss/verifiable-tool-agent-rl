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


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze paired runs by target-order position")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "paired_comparison": compare_pair(args.baseline, args.candidate),
        "position_analysis": analyze_position_runs(
            load_tasks(args.tasks),
            read_jsonl(args.baseline),
            read_jsonl(args.candidate),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
