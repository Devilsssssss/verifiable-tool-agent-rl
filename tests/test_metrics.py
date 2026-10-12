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
    assert "terminal_goal_completion" not in summary["overall"]


def test_recovery_metrics_use_format_error_episodes_as_denominator() -> None:
    base = {
        "family": "cancel",
        "success": False,
        "policy_compliant": False,
        "valid_tool_rate": 0.5,
        "steps": 6,
        "shaped": -0.5,
    }
    rows = [
        {
            **base,
            "terminal_goal_satisfied": True,
            "format_recovery_success": True,
            "violations": ["invalid_tool_call"],
        },
        {
            **base,
            "terminal_goal_satisfied": True,
            "format_recovery_success": False,
            "violations": ["invalid_tool_call", "unobserved_identifier"],
        },
        {
            **base,
            "success": True,
            "policy_compliant": True,
            "terminal_goal_satisfied": True,
            "format_recovery_success": False,
            "violations": [],
        },
    ]
    summary = summarize_reports(rows)["overall"]
    assert summary["task_success"] == 1 / 3
    assert summary["terminal_goal_completion"] == 1.0
    assert summary["format_error_tasks"] == 2
    assert summary["format_recovery_successes"] == 1
    assert summary["format_recovery_rate"] == 0.5


def test_recovery_rate_is_undefined_without_format_errors() -> None:
    rows = [
        {
            "family": "cancel",
            "success": True,
            "policy_compliant": True,
            "valid_tool_rate": 1.0,
            "steps": 5,
            "shaped": 1.4,
            "terminal_goal_satisfied": True,
            "format_recovery_success": False,
            "violations": [],
        }
    ]
    assert summarize_reports(rows)["overall"]["format_recovery_rate"] is None
