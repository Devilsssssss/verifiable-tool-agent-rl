from __future__ import annotations

from collections import defaultdict
from typing import Any


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("cannot summarize empty reports")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        grouped[str(report["family"])].append(report)

    def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        return {
            "tasks": count,
            "task_success": sum(bool(row["success"]) for row in rows) / count,
            "policy_compliance": sum(bool(row["policy_compliant"]) for row in rows) / count,
            "mean_valid_tool_rate": sum(float(row["valid_tool_rate"]) for row in rows)
            / count,
            "mean_steps": sum(int(row["steps"]) for row in rows) / count,
            "mean_progress": sum(float(row.get("progress", 0.0)) for row in rows) / count,
            "mean_shaped_reward": sum(float(row["shaped"]) for row in rows) / count,
        }

    return {
        "overall": summarize_group(reports),
        "by_family": {
            family: summarize_group(rows) for family, rows in sorted(grouped.items())
        },
    }

