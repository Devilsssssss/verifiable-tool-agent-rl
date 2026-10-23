from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.io import read_jsonl
from toolrl.prompts import REASON_CODE_INPUT_CONTRACT

ALLOWED_CODES = {
    "shipped_order_cannot_cancel",
    "return_window_expired",
}
ESCALATION_FAMILIES = {
    "cancel_shipped": "shipped_order_cannot_cancel",
    "return_expired": "return_window_expired",
}


def _audit_rows(path: Path) -> dict[str, int]:
    rows = read_jsonl(path)
    failures: list[str] = []
    prompt_rows = 0
    escalation_rows = 0
    for row in rows:
        task_id = str(row.get("task_id", ""))
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            failures.append(f"{task_id}:missing_messages")
            continue
        system = messages[0].get("content", "") if isinstance(messages[0], dict) else ""
        if REASON_CODE_INPUT_CONTRACT not in system and "reason_code" not in system:
            failures.append(f"{task_id}:missing_reason_code_prompt")
        else:
            prompt_rows += 1
        family = str(row.get("family", ""))
        expected = ESCALATION_FAMILIES.get(family)
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls", []):
                function = call.get("function", {}) if isinstance(call, dict) else {}
                if function.get("name") != "escalate_to_human":
                    continue
                escalation_rows += 1
                arguments = function.get("arguments", {})
                code = arguments.get("reason_code") if isinstance(arguments, dict) else None
                if code not in ALLOWED_CODES:
                    failures.append(f"{task_id}:invalid_reason_code:{code}")
                if expected is not None and code != expected:
                    failures.append(f"{task_id}:wrong_reason_code:{code}!={expected}")
    if failures:
        raise ValueError(f"{path}: {failures[:8]}")
    return {
        "rows": len(rows),
        "prompt_contract_rows": prompt_rows,
        "escalation_calls": escalation_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit reason-code-v1 SFT rows without model dependencies")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {str(path): _audit_rows(path) for path in args.data}
    summary = {"input_contract": REASON_CODE_INPUT_CONTRACT, "files": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
