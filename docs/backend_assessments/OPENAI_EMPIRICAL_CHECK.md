# OMNIAMIND — OpenAI Empirical Check

**Author:** Massimiliano Brighindi  
**Project:** MB-X.01  
**Backend target:** OpenAI API  
**Status:** Pending empirical verification

---

## Purpose

This document defines the empirical checks required to determine whether OpenAI is a usable Level 1 backend for the first OMNIAMIND experiment.

The goal is not to evaluate the provider in general.

The goal is narrower:

- verify whether OpenAI exposes enough Level 1 candidate data
- verify whether first Split / Bifurcation Pressure proxies are computable
- move the backend classification from **PROVISIONAL** to one of:
  - **ACCEPT**
  - **PROVISIONAL**
  - **REJECT**

This is an execution-facing verification document.

---

## Position in the current OMNIAMIND path

The current OMNIAMIND first path is already constrained as follows:

- **Access level:** Level 1
- **First target phenomena:** Split and Bifurcation Pressure
- **Minimum required inputs:** stepwise candidate set `C_t` and score-like values `P_t(c_i)`
- **Current backend status:** OpenAI classified as **PROVISIONAL** pending empirical verification

This document exists to resolve that pending status.

---

## Scope

This check does not attempt to validate OMNIAMIND as a whole.

It only asks whether OpenAI is operationally adequate for the first Level 1 proxy experiment.

The result of this document is backend classification only.

---

## Required test conditions

Before running the checks, the test record must declare explicitly:

1. exact OpenAI endpoint used
2. exact model used
3. request mode used
4. whether streaming is enabled or disabled
5. candidate exposure parameters used
6. decoding parameters used
7. number of repeated runs performed

No empirical result is interpretable without these seven declarations.

---

## Check 1 — Stepwise Coverage

### Question

Does the backend provide candidate-level score data for each generation step across the emitted sequence?

### Why it matters

Without stepwise candidate exposure, there is no usable Level 1 trajectory for OMNIAMIND.

### Method

Run a minimal controlled generation with candidate exposure enabled.

Inspect whether candidate data are returned for each emitted token position.

### PASS condition

Candidate-level data are available for all generated steps needed for sequence-level proxy computation.

### DEGRADED condition

Candidate-level data are available only for part of the sequence, or only under restricted output conditions.

### FAIL condition

Candidate-level data are absent, or available only for an unusably small subset of steps.

### Criticality

```text
CRITICAL

If this check fails, OpenAI cannot serve as the first Level 1 backend.


---

Check 2 — Effective k-Depth

Question

What is the actual maximum usable candidate depth per generation step?

Why it matters

If k is too shallow, Split and dispersion proxies may be artificially weakened by backend truncation.

Method

Request the maximum available candidate depth supported by the API configuration being tested.

Measure the actual number of distinct candidate items returned per step.

PASS condition

The backend returns a candidate depth sufficient for first-pass proxy computation and does so consistently across steps.

DEGRADED condition

Candidate depth is present but shallow, inconsistent, or reduced on part of the trace.

FAIL condition

Candidate depth is too shallow for meaningful first-pass Level 1 analysis, or cannot be verified clearly.

Criticality

CRITICAL

If this check fails, OpenAI should not be the first OMNIAMIND backend.


---

Check 3 — Score Semantics

Question

Are the exposed candidate scores stable and interpretable enough for Level 1 proxy use?

Why it matters

Split and Bifurcation Pressure require score-like values whose meaning is explicit enough to support comparison across steps.

Method

Inspect returned score fields and confirm:

score type

score range

score consistency across repeated calls

whether score interpretation is explicit at the API level


PASS condition

Score semantics are explicit and stable enough for proxy computation.

DEGRADED condition

Scores are present but partially constrained, transformed, or only partly documented.

FAIL condition

Scores are too opaque, inconsistent, or structurally unusable for first-pass proxy work.

Criticality

NON-CRITICAL BUT LOAD-BEARING

This check alone may not force REJECT, but severe opacity weakens interpretability.


---

Check 4 — Repeatability Under Controlled Settings

Question

Can repeated calls under matched settings produce sufficiently stable candidate traces for comparison?

Why it matters

OMNIAMIND does not require perfect determinism, but it does require that repeated runs under controlled settings remain interpretable.

Method

Run the same prompt multiple times under the closest available controlled settings.

Inspect:

token sequence stability

candidate-set stability

score stability

trace alignment feasibility


PASS condition

Repeated runs are stable enough for stepwise comparison and proxy aggregation.

DEGRADED condition

Repeated runs show limited drift but remain usable with caution.

FAIL condition

Repeated runs vary so strongly that candidate-level proxy interpretation collapses.

Criticality

LOAD-BEARING

A FAIL here may still permit limited exploratory use, but weakens backend suitability sharply.


---

Check 5 — Truncation / Filtering Observability

Question

Is there visible evidence that backend-side truncation, filtering, or candidate suppression materially alters the exposed candidate structure?

Why it matters

OMNIAMIND must not confuse provider-side candidate shaping with model-side structural dynamics.

Method

Inspect candidate exposure across several prompt types, including cases with higher expected competition or ambiguity.

Look for signs such as:

systematic clipping of lower-ranked candidates

suspiciously uniform candidate exposure

abrupt candidate suppression patterns

score patterns incompatible with natural competition structure


PASS condition

Candidate exposure behaves consistently enough that truncation/filtering does not appear to dominate the observed signal.

DEGRADED condition

Some truncation or shaping is visible, but candidate-level signal remains usable with explicit caveats.

FAIL condition

Filtering or truncation appears strong enough that Level 1 proxies would primarily measure provider-side shaping rather than usable candidate competition.

Criticality

LOAD-BEARING

A FAIL here strongly pushes classification toward REJECT or at least non-use for first-path work.


---

Check 6 — Operational Consistency

Question

Can the backend support enough repeated calls for first-pass sequence-level proxy testing without data degradation?

Why it matters

Even if candidate data exist, unstable response behavior can make the first experiment non-reproducible.

Method

Run a small batch of structured cases under matched settings.

Inspect:

latency stability

missing candidate fields

stepwise trace integrity

response consistency across the batch


PASS condition

The backend supports a small but real first-pass experiment with stable data integrity.

DEGRADED condition

The backend is usable, but rate limits, trace degradation, or intermittent missing fields reduce confidence.

FAIL condition

Operational instability prevents even small-scale repeatable Level 1 testing.

Criticality

NON-CRITICAL BUT IMPORTANT

This check does not define observability by itself, but it affects actual usability.


---

Execution rules

1. No wrapper layer

Use raw API calls or an official SDK only.

No orchestration layer should sit between the request and the observed candidate data.

2. Fixed test record

Every run must record:

endpoint

model

request payload

candidate exposure parameters

decoding parameters

output length constraint

timestamp of run


3. Same prompt set for repeated checks

Repeated checks should use the same small controlled prompt set.

4. Explicit status per check

Each check must be labeled as:

PASS

DEGRADED

FAIL


No narrative-only conclusion is allowed.


---

Kill-switch rule

Immediate backend rejection should occur if either of the following is true:

Check 1 (Stepwise Coverage) = FAIL

Check 2 (Effective k-Depth) = FAIL


Reason:

Without stepwise candidate coverage and adequate candidate depth, the first OMNIAMIND Level 1 path is not operational.


---

Classification logic

ACCEPT

Use ACCEPT only if:

both critical checks PASS

no other check FAILS

degraded items, if any, do not materially invalidate Level 1 proxy interpretation


PROVISIONAL

Use PROVISIONAL if:

both critical checks PASS

one or more non-critical checks are DEGRADED

candidate exposure is usable, but interpretability remains partially limited


REJECT

Use REJECT if:

either critical check FAILS

or backend-side shaping / instability makes first-path proxy interpretation non-credible



---

Minimum reporting block

The final empirical record must include:

1. backend name


2. endpoint used


3. model used


4. candidate type observed


5. score type observed


6. effective maximum k


7. stepwise coverage result


8. repeatability result


9. filtering / truncation observations


10. operational consistency result


11. final classification



Without these eleven items, the backend assessment is incomplete.


---

Consequence of failure

If OpenAI is classified as REJECT, the next correct step is:

docs/backend_assessments/LOCAL_CHECK.md

No intermediate abstraction layer should be inserted before that fallback assessment.


---

Minimal conclusion

This document exists to determine whether OpenAI can serve as the first real Level 1 backend for OMNIAMIND.

It does not test theory in general. It tests backend adequacy.

The result must be one of:

ACCEPT

PROVISIONAL

REJECT


based only on empirical observability and first-pass proxy usability.

