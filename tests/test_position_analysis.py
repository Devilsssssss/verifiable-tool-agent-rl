from toolrl.generator import generate_tasks
from toolrl.position_analysis import analyze_position_runs
from toolrl.position_shift import create_balanced_position_probe, target_order_position


def test_position_analysis_reports_stratified_paired_outcomes() -> None:
    tasks = create_balanced_position_probe(generate_tasks({"test": 20}, seed=46), seed=47)
    baseline = []
    candidate = []
    for index, task in enumerate(tasks):
        common = {"task_id": task.task_id, "family": task.family}
        baseline.append({**common, "success": False})
        candidate.append(
            {
                **common,
                "success": target_order_position(task) == 1 or index == 0,
            }
        )

    report = analyze_position_runs(tasks, baseline, candidate)

    assert report["strata"]["overall"]["tasks"] == 20
    assert report["strata"]["position_1"]["tasks"] == 10
    assert report["strata"]["position_2"]["tasks"] == 10
    assert report["strata"]["position_1"]["candidate_success_rate"] == 1.0
    assert report["position_effect"]["baseline_position_2_minus_1"] == 0.0
    assert "cancel_processing/position_1" in report["strata"]
