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
        summary = {
            "tasks": count,
            "task_success": sum(bool(row["success"]) for row in rows) / count,
            "policy_compliance": sum(bool(row["policy_compliant"]) for row in rows) / count,
            "mean_valid_tool_rate": sum(float(row["valid_tool_rate"]) for row in rows) / count,
            "mean_steps": sum(int(row["steps"]) for row in rows) / count,
            "mean_progress": sum(float(row.get("progress", 0.0)) for row in rows) / count,
            "mean_shaped_reward": sum(float(row["shaped"]) for row in rows) / count,
        }
        # Legacy artifacts lack terminal-state diagnostics. Do not invent them.
        if all("terminal_goal_satisfied" in row for row in rows):
            error_tasks = sum("invalid_tool_call" in row["violations"] for row in rows)
            recovered = sum(bool(row["format_recovery_success"]) for row in rows)
            summary.update(
                {
                    "terminal_goal_completion": sum(
                        bool(row["terminal_goal_satisfied"]) for row in rows
                    )
                    / count,
                    "format_error_tasks": error_tasks,
                    "format_recovery_successes": recovered,
                    "format_recovery_rate": recovered / error_tasks if error_tasks else None,
                }
            )
        return summary

    return {
        "overall": summarize_group(reports),
        "by_family": {family: summarize_group(rows) for family, rows in sorted(grouped.items())},
    }
