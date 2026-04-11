# OMNIAMIND — Backend Assessment: OpenAI

**Author:** Massimiliano Brighindi  
**Project:** MB-X.01  
**Assessment target:** OpenAI API  
**Assessment purpose:** First Level 1 eligibility check for OMNIAMIND Split / Bifurcation Pressure proxies

---

## Purpose

This document applies the OMNIAMIND backend eligibility checklist to OpenAI as a concrete candidate backend.

The goal is not to judge OpenAI in general.

The goal is narrower:

- determine whether OpenAI is acceptable for the first OMNIAMIND Level 1 proxy experiment
- identify which observability constraints are satisfied
- identify which constraints remain partial, degraded, or unknown
- decide whether OpenAI should be classified as ACCEPT, PROVISIONAL, or REJECT for this first path

---

## Scope note

This assessment is backend-specific and time-sensitive.

It does not define permanent truth about the provider.
It only records whether the provider appears usable for the current OMNIAMIND first experiment path.

If provider behavior, API surface, or exposure rules change, this document must be updated.

---

## Current OMNIAMIND requirement baseline

The first OMNIAMIND experiment currently requires:

- **Access level:** Level 1
- **First targets:** Split and Bifurcation Pressure
- **Minimum inputs:** stepwise candidate set `C_t` and score-like values `P_t(c_i)`
- **Experimental role:** first falsifiable proxy path only

This means OpenAI is acceptable only if it exposes enough candidate-level information to make those first proxies computable without hidden assumptions.

---

## Section A — Candidate visibility

### A1. Does the backend expose candidate items at each generation step?

**Assessment:** PARTIAL

**Reason:**
OpenAI can expose token-level candidate information in some configurations through logprob-related fields, but this is not equivalent to full candidate-trajectory exposure or explicit branch traces.

**Implication:**
Potentially usable for Level 1 token/candidate exposure, but limited.

---

### A2. Is the identity of each candidate item explicit?

**Assessment:** YES

**Reason:**
When candidate-level token exposure is available, the candidate identity is explicit at the token level.

**Implication:**
This satisfies the minimum identity condition, as long as the experiment explicitly declares that `c_i` means token candidate rather than continuation or branch.

---

### A3. Is the maximum available `k` explicit?

**Assessment:** PARTIAL

**Reason:**
The API may expose a bounded top-logprob candidate set, but the effective cap must be verified in practice for the exact endpoint/model used.

**Implication:**
Usable only if the actual maximum `k` is recorded empirically and reported.

---

## Section B — Score visibility

### B1. Does the backend expose score-like values for each candidate?

**Assessment:** YES

**Reason:**
OpenAI can expose logprob-like values for candidate tokens in supported configurations.

**Implication:**
This satisfies the minimum score visibility requirement for first Level 1 proxy attempts.

---

### B2. Is the score type explicit?

**Assessment:** YES

**Reason:**
If exposed, the score type is logprob-like rather than raw probability unless transformed by the user afterward.

**Implication:**
The experiment can remain interpretable if it reports score handling explicitly.

---

### B3. Are score truncation or filtering rules explicit?

**Assessment:** PARTIAL

**Reason:**
The provider may expose only top candidates and may apply backend-side constraints that are not fully transparent at the structural level.

**Implication:**
OpenAI remains usable only with explicit acknowledgment that observed candidate scores may be truncated and possibly filtered.

---

## Section C — Stepwise trace quality

### C1. Are candidate data available for each generation step?

**Assessment:** PARTIAL

**Reason:**
This must be verified on the exact API path used. The critical question is whether logprob candidate information is returned step-by-step across the generated sequence in the chosen setup.

**Implication:**
If stepwise trace coverage is incomplete, OpenAI weakens as a first OMNIAMIND target.

---

### C2. Are traces complete or truncated?

**Assessment:** PARTIAL

**Reason:**
OpenAI candidate exposure is likely partial rather than exhaustive.

**Implication:**
This does not automatically reject the backend, but it reduces signal completeness and must be reported.

---

### C3. Can candidate traces be aligned across repeated runs?

**Assessment:** PARTIAL

**Reason:**
Alignment may be possible under controlled settings, but backend variability and hidden provider-side adaptations may reduce exact comparability.

**Implication:**
Repeated-run analysis is possible only with caution.

---

## Section D — Reproducibility

### D1. Can the backend be run under controlled decoding settings?

**Assessment:** YES

**Reason:**
OpenAI generally exposes major decoding controls needed for a first-pass experiment.

**Implication:**
This satisfies a key first-proxy requirement.

---

### D2. Can the backend be tested at or near deterministic settings?

**Assessment:** PARTIAL

**Reason:**
Near-deterministic decoding may be possible, but exact determinism is not guaranteed in the same sense as a fully local controlled runtime.

**Implication:**
Useful for first-pass testing, but not equivalent to strict deterministic replay.

---

### D3. Are repeated runs feasible under comparable settings?

**Assessment:** YES

**Reason:**
Repeated calls can be made under controlled user-side settings, subject to cost/rate limits.

**Implication:**
This is sufficient for an initial Level 1 attempt.

---

## Section E — Rawness of exposure

### E1. Are candidate data exposed directly rather than through an orchestration wrapper?

**Assessment:** YES

**Reason:**
If used directly through the provider API, OpenAI avoids wrapper-layer ambiguity.

**Implication:**
This is preferable for first backend assessment.

---

### E2. Are backend-side transformations known?

**Assessment:** PARTIAL

**Reason:**
Some exposure constraints are known, but provider-side filtering, truncation, or hidden normalization may not be fully transparent.

**Implication:**
Interpretability is degraded relative to a local backend.

---

### E3. Is the provider likely to alter candidate exposure dynamically across requests?

**Assessment:** UNKNOWN

**Reason:**
This cannot be assumed absent without direct empirical testing.

**Implication:**
This uncertainty weakens OpenAI as a fully trusted structural source.

---

## Preliminary eligibility result

### Result: PROVISIONAL

OpenAI is not a clean ACCEPT for the first OMNIAMIND experiment.

But it is also not an automatic REJECT.

It is best classified as:

```text
PROVISIONAL

Why PROVISIONAL

OpenAI appears sufficient for a first constrained Level 1 attempt because:

candidate identity can be explicit at token level

score-like values can be exposed

repeated runs are feasible

operational startup is fast


But OpenAI remains limited because:

effective k must be verified

stepwise trace completeness must be verified

candidate exposure is truncated

provider-side filtering / transformation is not fully transparent

exact repeatability is weaker than with local direct control



---

What must be verified before actual use

Before OpenAI can be used for the first real OMNIAMIND proxy run, the following items must be checked empirically:

1. exact endpoint/model to be used


2. whether stepwise logprob candidate data are returned for all generated steps


3. maximum practical k


4. exact score field semantics


5. whether candidate exposure changes across repeated requests under matched settings


6. whether output-side or safety-side filtering visibly alters candidate traces



Without these six checks, the backend remains only provisionally acceptable.


---

Practical meaning for OMNIAMIND

If OpenAI passes the six empirical checks above with acceptable clarity, it becomes the fastest usable backend for the first Level 1 Split / Bifurcation Pressure experiment.

If one or more checks fail critically, the correct fallback is:

Local direct model backend

not a wrapper layer.


---

Current classification summary

Backend name: OpenAI API

Candidate type exposed: token-level candidates, if logprob exposure is enabled and supported

Score type exposed: logprob-like values

Maximum usable k: to be verified empirically

Stepwise trace condition: to be verified empirically

Reproducibility condition: partial-to-good under controlled settings

Known filtering / truncation / transformation: partial / not fully transparent

Final classification: PROVISIONAL



---

Minimal conclusion

OpenAI is a valid first backend candidate only as a provisional Level 1 target.

It is acceptable for a first small-scale OMNIAMIND attempt if and only if stepwise candidate exposure, effective k, and score semantics are confirmed empirically.

If those conditions fail, OpenAI should not be used as the first definitive backend for OMNIAMIND.

