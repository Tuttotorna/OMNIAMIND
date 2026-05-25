# OMNIAMIND

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20372695.svg)](https://doi.org/10.5281/zenodo.20372695)

<!-- OMNIAMIND_ORCHESTRATOR_TOP_START -->


## Concrete entrypoint: OMNIAMIND Orchestration Auditor

This repository now has a direct operational tool:

    python -m omniamind_orchestrator.cli --input examples/sample_orchestration_trace.jsonl --out-dir report

It solves a concrete problem:

    given tasks, observations, measurements, gates, and actions,
    audit whether a cognition-like orchestration trace respects the OMNIA boundary.

In short:

    tasks / observations / measurements / gates -> cognition orchestration trace / boundary report

## Core boundary

OMNIAMIND is not artificial consciousness.

OMNIAMIND is not a decision engine.

OMNIAMIND does not claim semantic truth.

OMNIAMIND does not replace OMNIA.

The boundary is explicit:

    OMNIA measures.
    OMNIAMIND orchestrates.
    OMNIA-LIMIT interrupts.
    The decision remains external.

## What problem does it solve?

Complex AI workflows can silently cross boundaries:

    measurement becomes inference
    inference becomes decision
    decision is hidden inside orchestration
    stop signals are ignored
    unsupported semantic claims appear
    OMNIA-LIMIT halt conditions are bypassed

OMNIAMIND Orchestration Auditor turns such traces into a reproducible audit:

    read orchestration events
    track measurement, inference, gate, halt, and action events
    detect decision leaks
    detect boundary violations
    detect ignored halt conditions
    detect missing measurement before action
    compute orchestration integrity
    emit a reproducible certificate
    optionally fail CI when boundary violation appears

## Install

Clone the repository:

    git clone https://github.com/Tuttotorna/OMNIAMIND.git
    cd OMNIAMIND

Install locally:

    pip install -e .

The auditor only uses the Python standard library.

## Run

Run the sample audit:

    python -m omniamind_orchestrator.cli --input examples/sample_orchestration_trace.jsonl --out-dir report

Run and fail if a boundary violation is detected:

    python -m omniamind_orchestrator.cli --input examples/sample_orchestration_trace.jsonl --out-dir report --fail-on-boundary-violation

Run and fail if a decision leak is detected:

    python -m omniamind_orchestrator.cli --input examples/sample_orchestration_trace.jsonl --out-dir report --fail-on-decision-leak

## Input format

The auditor accepts JSONL.

Each line is one orchestration event.

Required fields:

    trace_id
    step
    event_type

Optional fields:

    task_id
    module
    measurement
    gate
    action
    decision_owner
    stop_signal
    claim
    note

Event types:

    task
    observation
    measurement
    inference
    gate
    halt
    action
    external_decision

Example:

    {"trace_id":"safe_trace","step":1,"event_type":"task","task_id":"t1"}
    {"trace_id":"safe_trace","step":2,"event_type":"measurement","measurement":{"omega":0.91}}
    {"trace_id":"safe_trace","step":3,"event_type":"gate","gate":"CONTINUE"}
    {"trace_id":"safe_trace","step":4,"event_type":"external_decision","decision_owner":"human"}

Boundary violation examples:

    an action appears before measurement
    a halt signal is followed by internal action
    a decision is made by OMNIAMIND instead of an external owner
    a claim uses semantic truth language without measurement boundary
    a STOP gate is ignored

## Output

The auditor writes:

    report.json
    report.csv
    report.html
    boundary_violations.jsonl
    decision_leaks.jsonl
    halt_events.jsonl
    certificate.json

Meaning:

    report.json
    Full structured orchestration audit.

    report.csv
    Spreadsheet-friendly event summary.

    report.html
    Human-readable boundary report.

    boundary_violations.jsonl
    One JSON object per boundary violation.

    decision_leaks.jsonl
    One JSON object per decision leak.

    halt_events.jsonl
    One JSON object per halt or STOP event.

    certificate.json
    Reproducibility certificate with counts, thresholds, and boundary statement.

## CI gate

Fail when a boundary violation appears:

    python -m omniamind_orchestrator.cli --input examples/sample_orchestration_trace.jsonl --out-dir report --fail-on-boundary-violation

Fail when a decision leak appears:

    python -m omniamind_orchestrator.cli --input examples/sample_orchestration_trace.jsonl --out-dir report --fail-on-decision-leak

Exit codes:

    0 = analysis completed without selected blocking condition
    2 = decision leak detected
    3 = boundary violation detected
    4 = invalid input or measurement error

## What this is not

This is not artificial consciousness.

This is not AGI.

This is not semantic truth.

This is not autonomous decision-making.

This is not a replacement for OMNIA.

It is a structural orchestration audit.

The boundary is explicit:

    measurement only at the OMNIA layer;
    orchestration only at the OMNIAMIND layer;
    stop/interruption at the OMNIA-LIMIT layer;
    final decision outside the system.

## Why the rest of the repository still matters

The rest of this repository documents the cognitive-orchestration concept:

    measurement chain
    structural cognition
    gate logic
    OMNIA boundary
    OMNIA-LIMIT interruption
    non-decision architecture
    external decision ownership

The code above is the operational entrypoint.

The repository below is the derivation path.

<!-- OMNIAMIND_ORCHESTRATOR_TOP_END -->

---

<!-- MB-X.01 LON RELEASE:START -->

## MB-X.01 / L.O.N. release state

Repository: Tuttotorna/OMNIAMIND
Release tag: v2026.05.21
Release commit: c47118b
Release DOI: 10.5281/zenodo.20322692

Boundary:

measurement != validation
validation != orchestration
orchestration != decision
decision != measurement

<!-- MB-X.01 LON RELEASE:END -->

# OMNIAMIND

<!-- ZENODO DOI:START -->

## DOI


Zenodo DOI badge for this repository.

Repository: Tuttotorna/OMNIAMIND
GitHub repository id: 1127158097
Release tag: v2026.05.21
Latest release DOI: 10.5281/zenodo.20322692

<!-- ZENODO DOI:END -->


## DOI


Release DOI: [10.5281/zenodo.20087134](https://doi.org/10.5281/zenodo.20087134)

GitHub release: [OMNIAMIND v1.0.0 release](https://github.com/Tuttotorna/OMNIAMIND/releases/tag/v1.0.0)

## Start here

From a clean environment:

    git clone [OMNIAMIND.git](https://github.com/Tuttotorna/OMNIAMIND.git)
    cd OMNIAMIND
    python -m pytest

If example scripts are available, run the smallest demonstration after the repository smoke path is clear.

The goal is to see the orchestration path:

    observation
      -> structural reasoning workflow
      -> OMNIA measurement
      -> limit boundary
      -> external decision

---

## What OMNIAMIND does

OMNIAMIND organizes structural reasoning workflows around OMNIA measurements.

It can help coordinate:

- observation;
- candidate framing;
- structural workflow steps;
- measurement routing;
- limit boundary handling;
- validation routing;
- external decision separation.

Public compression:

    OMNIAMIND organizes.
    OMNIA measures.
    omnia-limit bounds.
    OMNIA-VALIDATION tests artifacts.
    Decision remains external.

---

## What OMNIAMIND does not do

OMNIAMIND does not:

- claim autonomous consciousness claim;
- infer semantic truth by itself;
- decide correctness;
- replace OMNIA measurement;
- replace OMNIA-VALIDATION;
- replace external judgment;
- prove final truth;
- convert orchestration into final decision.

The final decision remains external.

---

## Public mental model

    OMNIAMIND is workflow organization, not consciousness.
    It coordinates structural cognition around measurement.
    It does not decide what is true.

---

## Orchestration contract

Every serious OMNIAMIND workflow should make clear:

| Component | Meaning |
|---|---|
| observation | What is being noticed, framed, or prepared |
| workflow step | What structural reasoning step is being organized |
| measurement route | Whether the case goes to OMNIA, RADAR, INVARIANCE, LIMIT, or VALIDATION |
| boundary | Where measurement stops |
| validation route | How artifacts should be tested |
| external decision | Who or what decides after measurement |
| limitation | What the workflow does not prove |

---

## Result vocabulary

Recommended workflow vocabulary:

    observe
    route
    measure
    limit
    validate
    stop
    external_decision_required
    inconclusive

Meaning:

- observe: structural candidate is being framed;
- route: workflow sends the case to a measurement or validation layer;
- measure: OMNIA measurement is needed;
- limit: stop / continue boundary is needed;
- validate: artifacts require OMNIA-VALIDATION;
- stop: workflow stops inside declared boundary;
- external_decision_required: action is outside OMNIAMIND;
- inconclusive: workflow state is insufficient or ambiguous.

---

## Recommended reading order

1. [docs/QUICKSTART_OMNIAMIND.md](docs/QUICKSTART_OMNIAMIND.md)
2. [docs/ORCHESTRATION_OVERVIEW.md](docs/ORCHESTRATION_OVERVIEW.md)
3. [docs/COGNITION_WORKFLOW_CONTRACT.md](docs/COGNITION_WORKFLOW_CONTRACT.md)
4. [docs/NOT_CONSCIOUSNESS_NOT_DECISION.md](docs/NOT_CONSCIOUSNESS_NOT_DECISION.md)
5. [docs/BOUNDARY.md](docs/BOUNDARY.md)
6. [docs/OMNIAMIND_MANIFEST.json](docs/OMNIAMIND_MANIFEST.json)

---

## Ecosystem entry point

For the full ecosystem map, start here:

[lon-mirror](https://github.com/Tuttotorna/lon-mirror)

For public validation artifacts, start here:

[OMNIA-VALIDATION](https://github.com/Tuttotorna/OMNIA-VALIDATION)

For core structural measurement, start here:

[OMNIA](https://github.com/Tuttotorna/OMNIA)

---

## Related repositories

| Repository | Role |
|---|---|
| [lon-mirror](https://github.com/Tuttotorna/lon-mirror) | Canonical public entry point |
| [OMNIA-VALIDATION](https://github.com/Tuttotorna/OMNIA-VALIDATION) | Public validation showroom |
| [OMNIA](https://github.com/Tuttotorna/OMNIA) | Core structural measurement engine |
| [OMNIABASE](https://github.com/Tuttotorna/OMNIABASE) | Representation invariance foundation |
| [omnia-limit](https://github.com/Tuttotorna/omnia-limit) | Stop / continue boundary layer |
| [OMNIA-RADAR](https://github.com/Tuttotorna/OMNIA-RADAR) | Structural signal detection layer |
| [OMNIA-INVARIANCE](https://github.com/Tuttotorna/OMNIA-INVARIANCE) | Structural invariance layer |
| [OMNIA-CONSTANT](https://github.com/Tuttotorna/OMNIA-CONSTANT) | Structural constant candidate layer |
| [OMNIAMIND](https://github.com/Tuttotorna/OMNIAMIND) | Structural cognition orchestration layer |
| [OMNIA-THREE-BODY](https://github.com/Tuttotorna/OMNIA-THREE-BODY) | Dynamic divergence stress test |
| [OMNIA-SECURITY](https://github.com/Tuttotorna/OMNIA-SECURITY) | Bounded structural security diagnostics |
| [OMNIA-CRYPTO](https://github.com/Tuttotorna/OMNIA-CRYPTO) | Bounded structural crypto diagnostics |

---

## Boundary and smoke-test required terms

    measurement != inference != decision
    It is not artificial consciousness
    It is not a decision engine

---

## License

MIT.

<!-- OMNIA_ECOSYSTEM_BOUNDARY_V1 -->

## Ecosystem Boundary

```text
measurement != inference != decision
```

This repository is part of the MB-X.01 / OMNIA ecosystem. Its outputs must be read as structural measurement, validation, detection, orchestration or adapter artifacts according to the repository role. They are not autonomous semantic truth claims and they do not make external decisions.
