# OMNIAMIND — First Experiment Path

**Author:** Massimiliano Brighindi  
**Project:** MB-X.01

---

## Purpose

This document defines the first realistic experimental path for OMNIAMIND.

Its purpose is not to validate the full framework.

Its purpose is narrower:

- choose the first viable access level
- choose the first measurable phenomena
- define a minimal experimental scope
- prevent premature expansion into unavailable instrumentation or unfalsifiable claims

This is the entry path for OMNIAMIND experimentation.

---

## Position in the ecosystem

```text
Dual-Echo -> OMNIAMIND -> OMNIA -> OMNIA-LIMIT -> external decision layer

Within this chain:

OMNIAMIND measures pre-output structural dynamics

OMNIA measures post-hoc output stability under controlled transformations

OMNIA-LIMIT certifies structural saturation / stop conditions


The first OMNIAMIND experiment must remain compatible with this boundary.


---

Design rule

The first experiment must optimize for:

realism of access

clarity of signal

continuity with existing OMNIA work

low ambiguity of interpretation


It must not optimize for theoretical completeness.

A small falsifiable experiment is better than a broad speculative one.


---

Chosen access level

Level 1 — Token / Candidate Exposure

The first OMNIAMIND experiment should start at Access Level 1.

This means access to partial pre-output candidate information such as:

token logprobs

top-k token candidates

ranked continuation slices

beam candidates if available


Why Level 1

Level 1 is the first access regime that provides actual pre-output information without requiring privileged latent-state access or architecture-specific branch instrumentation.

It is the best balance between:

practical observability

structural relevance

implementation feasibility

portability across systems that expose candidate distributions


Why not Level 0 first

Level 0 remains useful for proxy diagnostics, but it is too indirect for the first OMNIAMIND experiment.

If the goal is to show that OMNIAMIND adds upstream structural visibility, then relying only on repeated external outputs is too weak as a first step.

Why not Level 2 or Level 3 first

Level 2 and Level 3 may later provide stronger structural diagnostics, but they depend on deeper visibility and stronger assumptions.

They are not the right entry point for a first falsifiable path.


---

Chosen phenomena

The first experiment should target:

Split

Bifurcation Pressure


Why Split

Split is the clearest first phenomenon because it can be approximated directly from visible candidate dispersion.

At Level 1, candidate branching becomes partially observable through:

spread among top-k candidates

relative branch separation

decay of single-path dominance


This makes split the most straightforward first target.

Why Bifurcation Pressure

Bifurcation pressure is the right companion target because it may become visible before a stable split fully appears.

At Level 1, it can be approximated through:

volatility in top candidate rankings

instability in relative candidate weights

rapid alternation among leading continuations


This makes it useful as a precursor signal.

Why not start with all six phenomena

The first experiment should not attempt full coverage.

Trying to measure all six at once would multiply ambiguity and weaken falsifiability.

The correct first step is to test whether:

a visible split signal exists

a precursor pressure signal exists

these signals vary meaningfully across more stable vs less stable cases



---

Experimental question

The first OMNIAMIND experiment asks:

Can Level 1 candidate exposure provide measurable upstream structural signals for split and bifurcation pressure that add information beyond post-hoc OMNIA alone?

This is the central question.

Not whether OMNIAMIND is complete. Not whether it universally works. Only whether these two pre-output signals are detectable and informative.


---

Minimal substrate assumption

The first experiment assumes access to one of the following:

token logprobs

top-k candidates per generation step

ranked candidate slices

beam paths if exposed


At minimum, the system must provide enough candidate-level visibility to reconstruct local competition among continuations.

If no such access exists, the experiment must be downgraded to Level 0 and explicitly labeled as such.


---

Minimal dataset strategy

The first experiment should reuse a narrow subset of existing OMNIA-compatible structured cases.

This preserves continuity with the current project boundary.

Preferred starting point:

structured outputs already used in the OMNIA runtime path

cases where post-hoc stability was already observable or testable

cases with clear success / fragility contrast


Why reuse existing OMNIA-related cases

This reduces noise.

It avoids creating a new dataset before testing whether OMNIAMIND signals are even measurable.

The first question is not dataset novelty. The first question is signal existence.

Initial dataset requirement

The initial dataset should contain at least two groups:

Group A: outputs later judged more structurally stable by OMNIA-like criteria

Group B: outputs later judged more structurally fragile or unstable


The goal is to test whether upstream Level 1 signals separate these groups before final emission.


---

Provisional signal sketches

The first experiment may use provisional non-final quantities such as:

Split signal

Possible proxies:

top-k candidate dispersion per step

average pairwise distance among leading candidate continuations

reduction in concentration of the top-ranked continuation

persistence of multi-branch competition across adjacent steps


Bifurcation pressure signal

Possible proxies:

volatility of top candidate ranking

rapid change in relative candidate probabilities

instability in dominance ordering across steps

local sensitivity of next-step candidate ordering under small perturbations


These quantities are not yet official metrics. They are only first experimental probes.


---

Comparison logic

The first experiment should compare at least two conditions:

Condition 1 — structurally cleaner cases

Cases that later appear more stable under OMNIA-like post-hoc analysis.

Condition 2 — structurally weaker cases

Cases that later appear more fragile, unstable, or collapse-prone under OMNIA-like post-hoc analysis.

The experiment then checks whether Level 1 pre-output signals show systematic differences between the two conditions.

The desired pattern is not perfection.

The desired pattern is directional usefulness.


---

Expected success criterion

The first experiment should be considered promising if at least one of the following is observed:

1. split-related quantities are consistently higher in structurally weaker cases


2. bifurcation-pressure quantities rise earlier or more strongly in structurally weaker cases


3. Level 1 signals provide separability not already obvious from final outputs alone


4. repeated runs show non-random directional behavior of these signals



This is enough for a first positive result.

The bar should remain modest.


---

Failure criterion

The first experiment should be considered negative or inconclusive if:

split proxies do not vary meaningfully across conditions

pressure proxies behave randomly or inconsistently

signals collapse under repetition

results depend entirely on one arbitrary implementation choice

no added visibility appears beyond post-hoc OMNIA


A negative result is acceptable.

The purpose of the first experiment is falsification pressure, not narrative preservation.


---

Reporting rule

Any result from this first path must report:

1. access level used


2. candidate substrate observed


3. dataset subset used


4. proxy quantities computed


5. comparison rule


6. observed limitations


7. whether the result is positive, negative, or inconclusive



No result should be reported without all seven items.


---

Non-goals

This first experiment does not aim to provide:

a complete OMNIAMIND metric suite

universal cross-model validity

latent-state geometry

explicit branch reconstruction

production abort logic

benchmark-grade final claims


Those belong to later stages, if justified.


---

Immediate consequence

If this first experiment shows no stable upstream signal at Level 1, then OMNIAMIND must either:

remain limited,

be reformulated,

or require deeper access levels for meaningful operation.


If it does show stable directional signal, then the next step becomes justified.


---

Next justified step after success

Only if the first experiment is promising, the next step should be:

define a small formal proxy metric for split

define a small formal proxy metric for bifurcation pressure

test repeatability

compare with OMNIA post-hoc outcomes

then consider expansion toward the remaining four phenomena


Not before.


---

Minimal conclusion

The first OMNIAMIND experiment should begin at Level 1.

It should target only:

Split

Bifurcation Pressure


It should reuse a narrow subset of existing structured OMNIA-compatible cases.

Its only goal is to determine whether pre-output candidate exposure provides measurable upstream structural signals that add information beyond post-hoc output analysis.

This is the correct first entry path.

Small. Falsifiable. Non-theatrical.