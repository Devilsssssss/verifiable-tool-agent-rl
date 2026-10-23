from __future__ import annotations

from dataclasses import replace

from .schemas import AgentTask

REASON_CODES = {
    "cancel_shipped": "shipped_order_cannot_cancel",
    "return_expired": "return_window_expired",
}


def strict_reason_code_task(task: AgentTask) -> AgentTask:
    """Expose the strict-v3 escalation code in the task goal and expert plan."""
    reason_code = REASON_CODES.get(task.family)
    if reason_code is None:
        return task
    goal = dict(task.goal)
    goal["reason_code"] = reason_code
    plan = []
    for step in task.expert_plan:
        updated = {"tool": step["tool"], "arguments": dict(step["arguments"])}
        if updated["tool"] == "escalate_to_human":
            updated["arguments"]["reason_code"] = reason_code
        plan.append(updated)
    return replace(task, goal=goal, expert_plan=tuple(plan))
