# OMNIAMIND Architecture

## Core architecture

```text
Input
  ↓
Reasoning / analysis layer
  ↓
OMNIAMIND
  ↓
OMNIA
  ↓
OMNIA-LIMIT
  ↓
External decision layer
```

---

## Layer roles

```text
OMNIAMIND = orchestration
OMNIA     = measurement
LIMIT     = interruption boundary
Decision  = external layer
```

OMNIAMIND organizes the cognitive pipeline.

OMNIA measures structural behavior.

OMNIA-LIMIT determines whether the process should stop, continue, retry, or escalate.

The final decision remains outside OMNIAMIND.

---

## Non-collapse rule

The system must not collapse orchestration into measurement or measurement into decision.

```text
organization != measurement != decision
measurement   != inference   != decision
```

---

## Correct interpretation

OMNIAMIND is not the layer that knows truth.

It is the layer that coordinates when and how a reasoning process is exposed to structural diagnostics.

A structural diagnostic may constrain a reasoning process.

It does not become final truth.
