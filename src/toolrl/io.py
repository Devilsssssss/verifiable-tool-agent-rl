from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schemas import AgentTask


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_tasks(path: Path) -> list[AgentTask]:
    return [AgentTask.from_dict(row) for row in read_jsonl(path)]
