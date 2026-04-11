# OMNIAMIND — First Backend Target

**Author:** Massimiliano Brighindi  
**Project:** MB-X.01

---

## Purpose

This document defines the first backend target for OMNIAMIND experimental work.

OMNIAMIND v2.0 begins with a Level 1 entry path.
That path is only real if a backend can expose the minimum candidate-level data required by the first proxy layer.

This document does not choose a final long-term infrastructure.

It chooses the first realistic backend class for the first falsifiable experiment.

---

## Position in the ecosystem

```text
Dual-Echo -> OMNIAMIND -> OMNIA -> OMNIA-LIMIT -> external decision layer

Within this chain:

OMNIAMIND requires pre-output candidate visibility

OMNIA operates on emitted output

OMNIA-LIMIT remains downstream as structural stop / saturation boundary


The first backend target must therefore be selected for candidate exposure, not for branding, not for benchmark scale, and not for theoretical elegance.


---

Constraint from previous documents

The current OMNIAMIND path is already fixed by prior constraints:

first access level: Level 1

first target phenomena: Split and Bifurcation Pressure

first experimental role: minimal falsifiable path

first proxy requirements: access to C_t and P_t(c_i) at each step


This means the backend must expose, at minimum:

top-k candidate items per step, or equivalent ranked candidate view

score-like values per candidate

stepwise generation trace sufficient for local proxy computation


If these conditions are absent, the backend is not a valid first target for Level 1.


---

Selection rule

The first backend target must optimize for four things only:

1. Candidate visibility

The backend must expose enough pre-output candidate information to compute first proxies.

2. Reproducibility

The backend must support repeated runs under controlled settings.

3. Operational simplicity

The backend must be usable without building a large new infrastructure before the first signal test.

4. Reporting clarity

The backend must make it possible to state clearly:

what C_t is

what P_t(c_i) is

what k is

how candidate traces are obtained


Any backend that fails these four constraints should be excluded from the first experiment path.


---

Candidate backend classes

Class A — Local direct model backend

Definition

A locally controlled model runtime with direct token-level or candidate-level access.

Typical properties:

local inference control

configurable decoding

direct access to top-k candidates or logits

repeatable parameter settings

stronger instrumentation options


Strengths

highest practical Level 1 control

possible future transition toward Level 2

explicit control over k

explicit control over temperature and decoding conditions

easier stepwise inspection


Weaknesses

requires local infrastructure

may be harder to run repeatedly on limited hardware

may introduce practical friction before the first experiment even starts


Use case

Best if the goal is maximum signal observability and the infrastructure is already available.


---

Class B — API backend with candidate exposure

Definition

A remote inference backend that exposes some candidate-level data through an API.

Typical properties:

easier operational startup

lower infrastructure burden

partial access to top candidates or logprob-like information

backend-specific constraints on k, trace depth, or score format


Strengths

fast to test

reproducible if parameters are exposed cleanly

suitable for first-pass experiments

easier to operationalize quickly than local full-control setups


Weaknesses

limited visibility compared with local backends

candidate exposure may be truncated

score formats may vary by provider

backend limits may constrain proxy quality


Use case

Best if the goal is the fastest route to a real Level 1 experiment without building custom inference infrastructure first.


---

Class C — Wrapped / orchestration backend

Definition

A backend accessed through an orchestration or middleware layer rather than direct model/runtime access.

Typical properties:

easier pipeline integration

possible abstraction over multiple providers

candidate information may be transformed, hidden, or normalized by the wrapper


Strengths

convenient for later integration

may simplify cross-backend comparison


Weaknesses

increased ambiguity about what C_t and P_t(c_i) actually mean

higher risk of hidden transformations

weaker reporting clarity

additional noise between model output and OMNIAMIND signal extraction


Use case

Not recommended as the first backend target unless the wrapper preserves raw candidate information transparently.


---

First-choice decision

The correct first backend target is:

Class B — API backend with candidate exposure

if and only if it exposes enough Level 1 data to compute the first proxies clearly.

Why this is the best first target:

it minimizes infrastructure delay

it keeps the experiment small

it allows a first falsifiable pass quickly

it is sufficient for Split and Bifurcation Pressure if top-k / logprob-like data are available

it preserves the principle that the first experiment should test signal existence before expanding complexity


This is the fastest non-theatrical path.


---

Fallback decision

If no available API backend exposes sufficient candidate-level information, the fallback is:

Class A — Local direct model backend

This becomes necessary when:

API access is too shallow

candidate traces are truncated beyond usefulness

score exposure is inconsistent

k cannot be controlled clearly

repeated runs are too constrained for meaningful proxy testing


In that case, local direct control becomes the first valid Level 1 path.


---

Exclusion rule

The first experiment should not start from Class C unless the wrapper layer can prove that it preserves raw candidate exposure without distortion.

Reason:

The first OMNIAMIND experiment must reduce ambiguity, not add another abstraction layer.


---

Minimum backend requirements

Any backend selected for the first experiment must declare explicitly:

1. candidate type used for C_t


2. score type used for P_t(c_i)


3. maximum available k


4. whether stepwise traces are complete or partial


5. whether repeated runs are controllable


6. whether backend-side truncation or filtering exists



Without these six items, the backend is not acceptable for the first proxy experiment.


---

Why backend choice matters structurally

The backend is not a tooling detail. It defines the effective observability boundary of OMNIAMIND.

Changing backend may change:

the meaning of candidates

the precision of split proxies

the reliability of pressure proxies

the repeatability of sequence-level aggregation

the comparison between upstream and downstream signals


For this reason, backend choice is part of the experimental specification, not an implementation footnote.


---

First recommended operational path

The first operational order should be:

1. identify one API backend that exposes candidate-level information clearly


2. verify actual availability of stepwise C_t and P_t(c_i)


3. determine the effective maximum usable k


4. test whether first proxies can be computed without hidden assumptions


5. only if this fails, move to a local direct backend



This order keeps the first experiment fast, constrained, and falsifiable.


---

Non-goals

This document does not decide:

the final production backend

the best long-term architecture

the Level 2 instrumentation path

multi-backend benchmarking

orchestration-layer integration

deployment strategy


It decides only the first realistic backend target for the first Level 1 OMNIAMIND experiment.


---

Minimal conclusion

For OMNIAMIND v2.0, the first backend target should be:

primary choice: API backend with sufficient candidate exposure

fallback choice: local direct model backend

excluded as first step: wrapped/orchestration backend unless raw candidate data are preserved transparently


The reason is simple:

the first experiment must test whether Split and Bifurcation Pressure produce usable upstream signal at Level 1 with minimal additional complexity.

Backend choice is therefore not secondary. It is the practical observability boundary of the first experiment.

