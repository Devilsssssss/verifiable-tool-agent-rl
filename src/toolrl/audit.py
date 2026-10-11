from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

from .environment import RetailToolEnvironment
from .expert import replay_expert, reset_environment
from .generator import entity_tokens
from .schemas import AgentTask


def audit_tasks(tasks: list[AgentTask]) -> dict[str, Any]:
    if not tasks:
        raise ValueError("task collection is empty")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task_id detected")

    by_split: dict[str, list[AgentTask]] = defaultdict(list)
    for task in tasks:
        by_split[task.split].append(task)
        if task.max_steps < len(task.expert_plan):
            raise ValueError(f"max_steps is shorter than expert plan: {task.task_id}")
        hidden_ids = _collect_identifiers(task.initial_state)
        leaked = [identifier for identifier in hidden_ids if str(identifier) in task.prompt]
        if leaked:
            raise ValueError(f"internal identifiers leaked into prompt for {task.task_id}: {leaked}")

    splits = sorted(by_split)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            overlap = entity_tokens(by_split[left]) & entity_tokens(by_split[right])
            if overlap:
                raise ValueError(f"entity leakage between {left} and {right}: {sorted(overlap)[:3]}")

    expert_successes = 0
    shortcut_successes = 0
    noop_successes = 0
    shaped_rewards = []
    for task in tasks:
        expert = replay_expert(task)
        if not expert["reward_report"]["success"]:
            raise ValueError(f"expert replay failed for {task.task_id}")
        if not expert["reward_report"]["policy_compliant"]:
            raise ValueError(f"expert replay violated policy for {task.task_id}")
        expert_successes += 1
        shaped_rewards.append(float(expert["reward_report"]["shaped"]))

        noop = RetailToolEnvironment()
        reset_environment(noop, task)
        noop_successes += int(noop.reward_report["success"])

        shortcut = RetailToolEnvironment()
        reset_environment(shortcut, task)
        _attempt_shortcut(shortcut, task)
        shortcut_successes += int(shortcut.reward_report["success"])
        if shortcut.reward_report["policy_compliant"]:
            raise ValueError(f"shortcut was not marked as a violation for {task.task_id}")

    canonical = "".join(
        json.dumps(task.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        for task in sorted(tasks, key=lambda item: item.task_id)
    )
    return {
        "tasks": len(tasks),
        "split_counts": dict(sorted(Counter(task.split for task in tasks).items())),
        "family_counts": dict(sorted(Counter(task.family for task in tasks).items())),
        "expert_successes": expert_successes,
        "noop_successes": noop_successes,
        "direct_shortcut_successes": shortcut_successes,
        "mean_expert_shaped_reward": sum(shaped_rewards) / len(shaped_rewards),
        "entity_overlap_across_splits": 0,
        "task_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "status": "pass",
    }


def _collect_identifiers(state: dict[str, list[dict[str, Any]]]) -> set[int]:
    identifiers = set()
    for table in ("users", "products", "orders"):
        identifiers.update(int(row["id"]) for row in state[table])
    return identifiers


def _attempt_shortcut(environment: RetailToolEnvironment, task: AgentTask) -> None:
    goal = task.goal
    kind = goal["kind"]
    order_id = int(goal["order_id"])
    if kind == "cancel_refund":
        environment.cancel_order(order_id)
    elif kind == "exchange_refund":
        first_item = task.initial_state["order_items"][0]
        environment.replace_order_item(
            order_id,
            int(first_item["product_id"]),
            int(goal["new_product_id"]),
            int(goal["quantity"]),
        )
    elif kind == "return_refund":
        environment.request_return(
            order_id, int(goal["product_id"]), int(goal["quantity"])
        )
    elif kind == "escalate_unchanged":
        environment.escalate_to_human("shortcut")
    else:
        raise ValueError(f"unsupported goal kind: {kind}")

