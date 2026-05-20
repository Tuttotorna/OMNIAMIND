# Cognition Workflow Contract

This document defines the public shape expected from OMNIAMIND workflow results.

The goal is clarity.

A reviewer should understand what was observed, how it was routed, where measurement happened, and where the workflow stopped.

---

## Workflow unit

A workflow result should contain:

| Component | Required | Meaning |
|---|---:|---|
| case_id | yes | Stable identifier for the workflow case |
| observation_ref | yes | Observation, trace, or candidate entering the workflow |
| workflow_step | yes | Current structural reasoning step |
| route | yes | Target layer or next operation |
| measurement_ref | preferred | OMNIA measurement artifact or reference |
| limit_ref | preferred | Stop / continue boundary artifact or reference |
| validation_ref | preferred | Validation artifact or route |
| workflow_result | yes | observe, route, measure, limit, validate, stop, external_decision_required, or inconclusive |
| limitation | yes | What the workflow does not prove |
| external_decision | yes | Explicit statement that final decision remains external |

---

## Minimal JSON shape

A minimal workflow artifact can use this shape:

    {{
      "case_id": "omniamind-example-001",
      "observation_ref": "path-or-description",
      "workflow_step": "declared step",
      "route": "OMNIA | RADAR | INVARIANCE | LIMIT | VALIDATION | external",
      "measurement_ref": null,
      "limit_ref": null,
      "validation_ref": null,
      "workflow_result": "observe | route | measure | limit | validate | stop | external_decision_required | inconclusive",
      "boundary": "measurement != inference != decision",
      "limitation": "What this workflow does not prove",
      "external_decision": "Decision remains external"
    }}

---

## Result vocabulary

Use a small vocabulary:

    observe
    route
    measure
    limit
    validate
    stop
    external_decision_required
    inconclusive

Meaning:

- observe: a candidate has entered the workflow;
- route: a target layer has been selected;
- measure: OMNIA measurement is needed;
- limit: stop / continue boundary is needed;
- validate: artifact validation is needed;
- stop: workflow stops inside declared boundary;
- external_decision_required: action belongs outside OMNIAMIND;
- inconclusive: workflow state is insufficient.

---

## No silent promotion

A workflow result must not silently become a final decision.

Routing is not truth.

Measurement request is not measurement.

Workflow organization is not consciousness.

