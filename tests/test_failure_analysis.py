import sys
from pathlib import Path

from toolrl.generator import generate_tasks
from toolrl.position_shift import create_balanced_position_probe

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_failure_deltas import analyze_failure_deltas


def test_failure_analysis_groups_deltas_and_records_failure_modes() -> None:
    tasks = create_balanced_position_probe(generate_tasks({"test": 20}, seed=46), seed=47)
    baseline = []
    candidate = []
    for index, task in enumerate(tasks):
        task_id = task.task_id
        baseline.append({"task_id": task_id, "success": index == 0})
        candidate.append(
            {
                "task_id": task_id,
                "success": index == 1,
                "terminal_goal_satisfied": index != 2,
                "target_order_inspected": index != 3,
                "violations": [] if index == 1 else ["return_window_closed"],
                "trace": [{"tool": "lookup_user"}],
            }
        )

    report = analyze_failure_deltas(tasks, baseline, candidate)

    assert report["overall"]["tasks"] == 20
    assert report["overall"]["gained_tasks"] == 1
    assert report["overall"]["lost_tasks"] == 1
    assert report["overall"]["candidate_failed_with_correct_terminal_state"] == 18
    assert report["overall"]["candidate_failed_without_target_inspection"] == 1
    assert "cancel_processing/position_1" in report["by_stratum"]
    assert report["failure_examples"]["gained"][0]["task_id"] == tasks[1].task_id
