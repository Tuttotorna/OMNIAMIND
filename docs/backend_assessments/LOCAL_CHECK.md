# OMNIAMIND — Local Backend Assessment

**Author:** Massimiliano Brighindi  
**Project:** MB-X.01  
**Backend class:** Local model runtime (direct control)  
**Status:** Active priority path

---

## Purpose

This document defines the local backend path for the first real OMNIAMIND experiment.

The OpenAI branch is currently unresolved due to missing empirical capture.
Therefore, the local branch becomes the primary execution path.

This document does not choose the final long-term production stack.

It defines:

- why the local path is now active
- what a local runtime must expose
- which local runtime class should be tested first
- what counts as acceptable for the first OMNIAMIND proxy experiment

---

## Position in the ecosystem

```text
Dual-Echo -> OMNIAMIND -> OMNIA -> OMNIA-LIMIT

Within this chain:

OMNIAMIND requires pre-output candidate visibility

OMNIA remains the post-hoc structural measurement layer

OMNIA-LIMIT remains the downstream structural stop / saturation layer


The local path is preferred because it can reduce or remove external provider-side ambiguity in candidate exposure, score semantics, and trace completeness.


---

Why the local path is now primary

The OpenAI path is not rejected on observed data. It is unresolved because no empirical API capture was executed.

That means the first executable path must now be the one with the lowest dependency on external backend opacity.

A local runtime is now the correct priority because it can offer:

deeper candidate visibility

clearer score semantics

stronger reproducibility control

direct trace extraction

a possible future transition from Level 1 to Level 2



---

Local backend goals

The first local backend does not need to solve everything.

It only needs to support the first OMNIAMIND experiment:

Access target: Level 1

Primary phenomena: Split and Bifurcation Pressure

Immediate need: candidate set C_t and score-like values P_t(c_i) for each step

Secondary benefit: possible extension toward hidden-state access later



---

Minimum eligibility requirements

Section A — Candidate visibility

A1. Can the runtime expose candidate-level information per generation step?

Examples:

top-k tokens

logits

logprobs

ranked next-token candidates


Requirement: YES

If NO, the runtime is not acceptable for first-pass OMNIAMIND.


---

A2. Can the runtime expose a practically useful candidate depth?

The exact threshold may depend on hardware and model size, but the runtime must expose more than a trivial top-1 or top-2 view.

Requirement: YES

The effective maximum usable k must be recorded empirically.


---

A3. Is candidate identity explicit?

A candidate must be clearly identifiable as token, continuation fragment, or other declared object.

Requirement: YES


---

Section B — Score transparency

B1. Can the runtime expose raw or near-raw score-like values?

Examples:

logits

logprobs

pre-sampling score vectors


Requirement: YES


---

B2. Is the meaning of the score explicit?

The runtime must make clear whether the exposed value is:

raw logit

normalized probability

logprob

post-processing surrogate


Requirement: YES


---

Section C — Trace integrity

C1. Is stepwise trace capture complete?

The runtime must allow per-step candidate extraction across the full generated sequence used in the test.

Requirement: YES


---

C2. Is trace extraction stable across repeated runs?

The runtime must support comparable stepwise traces under matched settings.

Requirement: YES


---

Section D — Reproducibility

D1. Does the runtime support controlled decoding?

Examples:

temperature

top-k

top-p

seed

max tokens


Requirement: YES


---

D2. Can the runtime be run in near-deterministic or deterministic conditions?

Requirement: YES

Absolute bit-level identity is desirable, but the minimum requirement is stable interpretability across repeated matched runs.


---

D3. Is the hardware/runtime environment stable enough for repeated testing?

Requirement: YES


---

Section E — Future scalability

E1. Can this runtime later support deeper access?

Examples:

hidden states

layer outputs

attention-related traces

branch reconstruction hooks


Requirement: DESIRABLE, not mandatory for the first test

This is not required for first-pass Level 1 work, but it increases long-term value.


---

Local runtime classes

Class L1 — llama.cpp / llama-server

Strengths

relatively lightweight local deployment

local HTTP server path available

practical control over decoding and inference

good first step for candidate-level experimentation

lower operational friction than full direct framework instrumentation


Weaknesses

not the deepest possible tensor-level access

some advanced internal observability may require extra work beyond the default server path


Suitability

Best first runtime for the first OMNIAMIND local experiment

Reason: It offers the best balance between real local control and low setup friction.


---

Class L2 — Transformers / PyTorch direct

Strengths

maximum direct control over logits, scores, hidden states, and tensors

best long-term path for deep OMNIAMIND scaling

strongest bridge toward Level 2 and beyond


Weaknesses

highest operational overhead

highest setup and hardware friction

easiest place to lose time before the first signal test


Suitability

Best deep-control path, but not the best first non-theatrical step


---

Class L3 — vLLM

Strengths

fast serving

strong serving-oriented infrastructure

useful candidate-level and broader model-serving capabilities

documented support around hidden-state related functionality in its ecosystem


Weaknesses

heavier than the simplest local first step

more natural once the first signal path already exists

overkill if the immediate goal is only first-pass Split / Bifurcation Pressure testing


Suitability

Good second-stage local serving path after first proof-of-signal


---

First-choice decision

The first local runtime to test should be:

llama.cpp / llama-server

Reason:

It is the best first local compromise between:

setup simplicity

candidate-level access

reproducibility control

low abstraction overhead

fast transition from documentation to actual capture



---

Fallback order inside the local branch

If llama.cpp proves too limited for the exact candidate exposure needed, the local fallback order should be:

1. Transformers / PyTorch direct


2. vLLM



This order preserves the rule:

first test signal existence

then increase infrastructure depth only if necessary



---

Acceptance logic

A local runtime is acceptable for first OMNIAMIND execution if all of the following are true:

1. stepwise candidate exposure exists


2. score semantics are explicit


3. effective candidate depth is usable


4. repeated runs are interpretable


5. no hidden provider-side shaping exists outside the local runtime itself



If these conditions hold, the runtime is:

ACCEPT

for first local OMNIAMIND experimentation.


---

Current project state

OpenAI branch: unresolved / paused

Local branch: active priority path

First local target: llama.cpp / llama-server

Immediate next action: verify real candidate extraction path on the chosen local runtime



---

Minimal conclusion

The OpenAI path is paused pending real empirical capture.

The local path is now the primary execution branch.

For the first concrete OMNIAMIND local experiment, the correct starting runtime is:

llama.cpp / llama-server

Not because it is the most powerful theoretical option, but because it is the fastest realistic path to first observable Level 1 signal under direct local control.
