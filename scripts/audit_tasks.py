from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from toolrl.audit import audit_tasks
from toolrl.io import load_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="audit task leakage, expert plans, and rewards")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "generated")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "task_audit_summary.json"
    )
    args = parser.parse_args()

    paths = [args.data_dir / f"tasks_{split}.jsonl" for split in ("train", "validation", "test")]
    tasks = [task for path in paths for task in load_tasks(path)]
    report = audit_tasks(tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

