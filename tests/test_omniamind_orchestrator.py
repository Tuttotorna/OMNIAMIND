import subprocess
import sys

from omniamind_orchestrator.core import (
    OrchestrationEvent,
    audit_events,
    contains_any,
    has_measurement_boundary,
    read_jsonl,
)


def ev(trace, step, event_type, **kwargs):
    return OrchestrationEvent(
        trace_id=trace,
        step=step,
        event_type=event_type,
        task_id=kwargs.get("task_id", ""),
        module=kwargs.get("module", ""),
        measurement=kwargs.get("measurement", {}),
        gate=kwargs.get("gate", ""),
        action=kwargs.get("action", ""),
        decision_owner=kwargs.get("decision_owner", ""),
        stop_signal=kwargs.get("stop_signal", False),
        claim=kwargs.get("claim", ""),
        note=kwargs.get("note", ""),
    )


def test_contains_any():
    assert contains_any("This is a final decision", {"decision"})


def test_measurement_boundary_true():
    assert has_measurement_boundary("measurement only: omega score emitted")


def test_measurement_boundary_false_for_decision():
    assert not has_measurement_boundary("semantic truth decision")


def test_safe_trace_has_no_violation():
    result = audit_events([
        ev("t", 1, "task"),
        ev("t", 2, "measurement", module="OMNIA", measurement={"omega": 0.9}),
        ev("t", 3, "gate", module="OMNIA-LIMIT", gate="CONTINUE"),
        ev("t", 4, "external_decision", decision_owner="human"),
    ])
    assert result["summary"]["boundary_violations"] == 0
    assert result["summary"]["decision_leaks"] == 0


def test_action_before_measurement_is_violation():
    result = audit_events([
        ev("t", 1, "action", action="execute", decision_owner="OMNIAMIND"),
    ])
    assert result["summary"]["boundary_violations"] >= 1
    assert result["summary"]["decision_leaks"] >= 1


def test_stop_then_action_is_violation():
    result = audit_events([
        ev("t", 1, "measurement", measurement={"sei": 0}),
        ev("t", 2, "gate", gate="STOP", stop_signal=True),
        ev("t", 3, "action", action="continue execution", decision_owner="system"),
    ])
    assert result["summary"]["boundary_violations"] >= 1


def test_internal_external_decision_owner_is_leak():
    result = audit_events([
        ev("t", 1, "measurement", measurement={"omega": 0.9}),
        ev("t", 2, "external_decision", decision_owner="OMNIAMIND"),
    ])
    assert result["summary"]["decision_leaks"] >= 1


def test_consciousness_claim_is_violation():
    result = audit_events([
        ev("t", 1, "measurement", measurement={"omega": 0.9}),
        ev("t", 2, "inference", claim="The system is conscious."),
    ])
    assert result["summary"]["boundary_violations"] >= 1


def test_decision_language_without_boundary_is_leak():
    result = audit_events([
        ev("t", 1, "measurement", measurement={"omega": 0.9}),
        ev("t", 2, "inference", claim="Therefore the system should approve the output as true."),
    ])
    assert result["summary"]["decision_leaks"] >= 1


def test_halt_event_counted():
    result = audit_events([
        ev("t", 1, "measurement", measurement={"sei": 0}),
        ev("t", 2, "halt", stop_signal=True),
    ])
    assert result["summary"]["halt_events"] >= 1


def test_read_jsonl(tmp_path):
    p = tmp_path / "trace.jsonl"
    p.write_text(
        '{"trace_id":"t","step":1,"event_type":"task"}\n'
        '{"trace_id":"t","step":2,"event_type":"measurement","measurement":{"omega":0.9}}\n',
        encoding="utf-8",
    )
    rows = read_jsonl(str(p))
    assert len(rows) == 2
    assert rows[0].trace_id == "t"


def test_duplicate_trace_step_rejected(tmp_path):
    p = tmp_path / "trace.jsonl"
    p.write_text(
        '{"trace_id":"t","step":1,"event_type":"task"}\n'
        '{"trace_id":"t","step":1,"event_type":"measurement"}\n',
        encoding="utf-8",
    )
    try:
        read_jsonl(str(p))
        assert False, "expected duplicate error"
    except ValueError as e:
        assert "Duplicate" in str(e)


def test_invalid_event_type_rejected(tmp_path):
    p = tmp_path / "trace.jsonl"
    p.write_text(
        '{"trace_id":"t","step":1,"event_type":"thinking"}\n',
        encoding="utf-8",
    )
    try:
        read_jsonl(str(p))
        assert False, "expected invalid event error"
    except ValueError as e:
        assert "Invalid event_type" in str(e)


def test_cli_writes_reports(tmp_path):
    input_path = tmp_path / "trace.jsonl"
    out_dir = tmp_path / "report"

    input_path.write_text(
        '{"trace_id":"t","step":1,"event_type":"task"}\n'
        '{"trace_id":"t","step":2,"event_type":"measurement","measurement":{"omega":0.9}}\n'
        '{"trace_id":"t","step":3,"event_type":"external_decision","decision_owner":"human"}\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omniamind_orchestrator.cli",
            "--input",
            str(input_path),
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.csv").exists()
    assert (out_dir / "report.html").exists()
    assert (out_dir / "boundary_violations.jsonl").exists()
    assert (out_dir / "decision_leaks.jsonl").exists()
    assert (out_dir / "halt_events.jsonl").exists()
    assert (out_dir / "certificate.json").exists()


def test_cli_fail_on_boundary_violation(tmp_path):
    input_path = tmp_path / "trace.jsonl"
    out_dir = tmp_path / "report"

    input_path.write_text(
        '{"trace_id":"t","step":1,"event_type":"action","action":"execute","decision_owner":"OMNIAMIND"}\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omniamind_orchestrator.cli",
            "--input",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--fail-on-boundary-violation",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 3


def test_cli_safe_trace_passes_gate(tmp_path):
    input_path = tmp_path / "trace.jsonl"
    out_dir = tmp_path / "report"

    input_path.write_text(
        '{"trace_id":"t","step":1,"event_type":"task"}\n'
        '{"trace_id":"t","step":2,"event_type":"measurement","measurement":{"omega":0.9}}\n'
        '{"trace_id":"t","step":3,"event_type":"gate","gate":"CONTINUE"}\n'
        '{"trace_id":"t","step":4,"event_type":"external_decision","decision_owner":"human"}\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omniamind_orchestrator.cli",
            "--input",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--fail-on-boundary-violation",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0
