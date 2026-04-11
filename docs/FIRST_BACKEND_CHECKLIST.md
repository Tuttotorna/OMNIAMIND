# OMNIAMIND — First Backend Checklist

**Author:** Massimiliano Brighindi  
**Project:** MB-X.01

---

## Purpose

This document defines the minimum eligibility checklist for selecting the first real backend for OMNIAMIND Level 1 experimentation.

It does not validate OMNIAMIND.
It does not compare providers for marketing purposes.
It does not define long-term infrastructure.

Its role is narrower:

- determine whether a backend is usable for the first proxy experiment
- exclude backends that do not expose sufficient candidate-level information
- force explicit reporting of observability constraints before experimentation begins

This is an eligibility checklist, not a benchmark.

---

## Position in the current path

The current OMNIAMIND path is already constrained by previous documents:

- **Access level:** Level 1
- **First target phenomena:** Split and Bifurcation Pressure
- **First operational layer:** provisional proxies only
- **First backend target:** API backend with sufficient candidate exposure, with local backend as fallback

This checklist exists to decide whether a concrete backend instance satisfies that path.

---

## Core rule

A backend is acceptable for the first OMNIAMIND experiment only if it makes the Level 1 proxy layer operational without hidden assumptions.

If the backend obscures candidate identity, score meaning, trace completeness, or reproducibility, it should be rejected for the first experiment path.

---

## Eligibility checklist

For each backend under consideration, answer the following items explicitly.

Use one of:

- **YES**
- **PARTIAL**
- **NO**
- **UNKNOWN**

No backend should be accepted on ambiguity alone.

---

## Section A — Candidate visibility

### A1. Does the backend expose candidate items at each generation step?

Examples:

- top-k next tokens
- ranked candidate continuations
- beam candidates

Requirement:

```text id="ynsqw8"
Minimum acceptable answer: YES or PARTIAL

If NO, the backend is not valid for Level 1.


---

A2. Is the identity of each candidate item explicit?

The backend must make clear what a candidate actually is.

Examples:

token

token string

continuation fragment

branch candidate


Requirement:

Minimum acceptable answer: YES

If NO or UNKNOWN, the backend is not acceptable.


---

A3. Is the maximum available k explicit?

The backend must state or reveal how many candidates can be retrieved per step.

Requirement:

Minimum acceptable answer: YES

If NO, proxy interpretation becomes underspecified.


---

Section B — Score visibility

B1. Does the backend expose score-like values for each candidate?

Examples:

probabilities

logprobs

normalized scores

rank-derived values


Requirement:

Minimum acceptable answer: YES or PARTIAL

If NO, Split and Bifurcation Pressure proxies become too weak for first-pass Level 1.


---

B2. Is the score type explicit?

The backend must make clear whether scores are:

probabilities

logprobs

normalized surrogates

other provider-specific values


Requirement:

Minimum acceptable answer: YES

If NO or UNKNOWN, score interpretation is not reliable enough.


---

B3. Are score truncation or filtering rules explicit?

Examples:

only top-n candidates exposed

suppressed low-probability candidates

safety filters changing candidate list

normalization after filtering


Requirement:

Minimum acceptable answer: YES or PARTIAL

If NO or UNKNOWN, the backend remains usable only with explicit caution.


---

Section C — Stepwise trace quality

C1. Are candidate data available for each generation step?

Requirement:

Minimum acceptable answer: YES

If candidate data are only available for the final token or for a partial suffix, the backend is weak for OMNIAMIND Level 1.


---

C2. Are traces complete or truncated?

The backend must declare whether stepwise candidate exposure is:

complete

partial

suffix-only

sampled intermittently


Requirement:

Minimum acceptable answer: YES

Meaning: the trace condition must be known, even if imperfect.


---

C3. Can candidate traces be aligned across repeated runs?

The backend should make it possible to compare stepwise trajectories across repeated executions under similar settings.

Requirement:

Minimum acceptable answer: YES or PARTIAL

If NO, repeatability analysis becomes severely limited.


---

Section D — Reproducibility

D1. Can the backend be run under controlled decoding settings?

Examples:

temperature

top-p

max tokens

stop conditions


Requirement:

Minimum acceptable answer: YES

If NO, proxy behavior is too confounded.


---

D2. Can the backend be tested at or near deterministic settings?

Examples:

temperature = 0

fixed seed, if available

stable decoding mode


Requirement:

Minimum acceptable answer: YES or PARTIAL

If NO, the backend may still be usable, but interpretation becomes weaker.


---

D3. Are repeated runs feasible under comparable settings?

Requirement:

Minimum acceptable answer: YES

If repeated runs are heavily rate-limited, unstable, or hidden behind adaptive behavior, first-pass experimentation becomes unreliable.


---

Section E — Rawness of exposure

E1. Are candidate data exposed directly rather than through an orchestration wrapper?

Requirement:

Preferred answer: YES

If NO, the wrapper must prove that candidate data are preserved transparently.


---

E2. Are backend-side transformations known?

Examples:

safety filtering

candidate suppression

rank clipping

score normalization

hidden reranking


Requirement:

Minimum acceptable answer: YES or PARTIAL

If NO or UNKNOWN, proxy interpretation must be flagged as degraded.


---

E3. Is the provider likely to alter candidate exposure dynamically across requests?

Requirement:

Preferred answer: NO

If YES or UNKNOWN, repeatability and comparability weaken.


---

Section F — Acceptance criteria

A backend is acceptable for the first OMNIAMIND experiment if all of the following are true:

1. candidate identity is explicit


2. score type is explicit


3. maximum k is explicit


4. stepwise candidate traces are available


5. controlled decoding settings are available


6. repeated runs are feasible


7. no critical ambiguity remains about whether the observed data are raw enough for Level 1 proxy computation



If any of items 1–6 fails with NO, the backend should be rejected as first target.

If item 7 remains unresolved, the backend may only be accepted as provisional.


---

Backend classification outcome

After completing the checklist, classify the backend as one of:

ACCEPT

Usable for first Level 1 proxy experiment.

PROVISIONAL

Usable only with explicit caveats and degraded interpretability.

REJECT

Not suitable for the first OMNIAMIND experiment.


---

Minimum reporting format

For each backend tested, report at least:

1. backend name


2. candidate type exposed


3. score type exposed


4. maximum usable k


5. stepwise trace condition


6. reproducibility condition


7. known filtering / truncation / transformation


8. final classification: ACCEPT / PROVISIONAL / REJECT



Without these eight items, backend choice remains underspecified.


---

Interpretation rule

This checklist does not determine whether a backend is "good" in general.

It determines only whether the backend is adequate for the first OMNIAMIND Level 1 experiment.

A backend may be strong for production use and still be unusable for OMNIAMIND. A backend may be limited overall and still be acceptable for this first falsifiable path.

The criterion is observability, not reputation.


---

Non-goals

This document does not decide:

the best long-term provider

the cheapest provider

the strongest general model

the final OMNIAMIND production stack

Level 2 or Level 3 instrumentation strategy


It only filters backend eligibility for the first Level 1 proxy test.


---

Minimal conclusion

The first OMNIAMIND experiment requires a backend that exposes enough candidate-level information to make Split and Bifurcation Pressure proxies computable without hidden assumptions.

This checklist exists to determine that eligibility before coding begins.

If a backend fails candidate visibility, score clarity, stepwise trace availability, or reproducibility, it should not be used as the first OMNIAMIND target.


