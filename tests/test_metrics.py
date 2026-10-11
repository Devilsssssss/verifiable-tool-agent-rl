from toolrl.metrics import summarize_reports


def test_summarize_reports_keeps_family_breakdown() -> None:
    reports = [
        {
            "family": "cancel",
            "success": True,
            "policy_compliant": True,
            "valid_tool_rate": 1.0,
            "steps": 4,
            "progress": 1.0,
            "shaped": 1.4,
        },
        {
            "family": "return",
            "success": False,
            "policy_compliant": True,
            "valid_tool_rate": 0.5,
            "steps": 2,
            "progress": 0.5,
            "shaped": 0.05,
        },
    ]

    summary = summarize_reports(reports)

    assert summary["overall"]["task_success"] == 0.5
    assert summary["overall"]["policy_compliance"] == 1.0
    assert summary["by_family"]["cancel"]["task_success"] == 1.0
    assert summary["by_family"]["return"]["task_success"] == 0.0
    assert summary["overall"]["mean_progress"] == 0.75

