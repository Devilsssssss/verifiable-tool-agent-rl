from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    family: str
    split: str
    prompt: str
    initial_state: dict[str, list[dict[str, Any]]]
    goal: dict[str, Any]
    expert_plan: tuple[dict[str, Any], ...]
    max_steps: int
    optimal_steps: int
    template_id: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["expert_plan"] = list(self.expert_plan)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> AgentTask:
        return cls(
            task_id=str(row["task_id"]),
            family=str(row["family"]),
            split=str(row["split"]),
            prompt=str(row["prompt"]),
            initial_state=dict(row["initial_state"]),
            goal=dict(row["goal"]),
            expert_plan=tuple(dict(step) for step in row["expert_plan"]),
            max_steps=int(row["max_steps"]),
            optimal_steps=int(row["optimal_steps"]),
            template_id=str(row["template_id"]),
        )

