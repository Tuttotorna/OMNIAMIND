import argparse
import sys
from pathlib import Path

from .core import audit_events, read_jsonl, write_all_reports


def main():
    parser = argparse.ArgumentParser(
        prog="omniamind-orchestrate-audit",
        description="Audit OMNIAMIND orchestration traces for boundary violations, decision leaks, and ignored halt conditions.",
    )

    parser.add_argument("--input", required=True, help="JSONL file with orchestration events.")
    parser.add_argument("--out-dir", default="omniamind_report", help="Output directory.")
    parser.add_argument("--fail-on-boundary-violation", action="store_true", help="Exit with code 3 if any boundary violation appears.")
    parser.add_argument("--fail-on-decision-leak", action="store_true", help="Exit with code 2 if decision leak appears, or 3 if boundary violation appears.")

    args = parser.parse_args()

    try:
        events = read_jsonl(args.input)
        result = audit_events(events)
        write_all_reports(args.out_dir, result)
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(4)

    s = result["summary"]

    print("")
    print("OMNIAMIND ORCHESTRATION AUDIT")
    print("=============================")
    print(f"input:                    {args.input}")
    print(f"total_traces:             {s['total_traces']}")
    print(f"total_events:             {s['total_events']}")
    print(f"clear_events:             {s['clear_events']}")
    print(f"halt_events:              {s['halt_events']}")
    print(f"decision_leaks:           {s['decision_leaks']}")
    print(f"boundary_violations:      {s['boundary_violations']}")
    print(f"clear_rate:               {s['clear_rate']:.6f}")
    print(f"decision_leak_rate:       {s['decision_leak_rate']:.6f}")
    print(f"boundary_violation_rate:  {s['boundary_violation_rate']:.6f}")
    print(f"min_integrity:            {s['min_orchestration_integrity']}")
    print(f"max_violation_score:      {s['max_violation_score']}")
    print(f"worst_trace_id:           {s['worst_trace_id']}")
    print(f"worst_step:               {s['worst_step']}")
    print("")
    print(f"WROTE: {Path(args.out_dir) / 'report.json'}")
    print(f"WROTE: {Path(args.out_dir) / 'report.csv'}")
    print(f"WROTE: {Path(args.out_dir) / 'report.html'}")
    print(f"WROTE: {Path(args.out_dir) / 'boundary_violations.jsonl'}")
    print(f"WROTE: {Path(args.out_dir) / 'decision_leaks.jsonl'}")
    print(f"WROTE: {Path(args.out_dir) / 'halt_events.jsonl'}")
    print(f"WROTE: {Path(args.out_dir) / 'certificate.json'}")
    print("")

    if args.fail_on_boundary_violation and s["boundary_violations"] > 0:
        sys.exit(3)

    if args.fail_on_decision_leak:
        if s["boundary_violations"] > 0:
            sys.exit(3)
        if s["decision_leaks"] > 0:
            sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
