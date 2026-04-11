# OMNIAMIND — Proxy Formalization v0

**Author:** Massimiliano Brighindi  
**Project:** MB-X.01  
**Status:** Synthetic / structural / non-semantic

---

## Purpose

This document defines the first formal proxy layer for OMNIAMIND.

Version `v0` is intentionally minimal.

It does not define final metrics.
It does not claim universal validity.
It does not depend on semantic interpretation.

Its purpose is narrower:

- define computable structural proxies
- keep the layer non-semantic
- provide a mathematical basis for synthetic testing
- support the first deterministic proxy runner

This document formalizes only two OMNIAMIND targets:

- **Split**
- **Bifurcation Pressure**

---

## Boundary conditions

The proxy layer must remain strictly structural.

It must not depend on:

- semantic similarity
- synonymy
- meaning
- truth
- correctness
- reasoning quality
- psychological interpretation

OMNIAMIND v0 measures only the geometry of candidate probability structure across steps.

---

## 1. Candidate set and local distribution

At generation step `t`, let:

```text
C_t = {c_1, c_2, ..., c_k}

be the set of top-k candidate items.

Each candidate c_i has an associated local probability:

P_t(c_i)

with local normalization:

sum_i P_t(c_i) = 1

in the observed candidate space for step t.

Interpretation

This is not the full model vocabulary. It is only the local candidate space exposed or selected for analysis.

The proxy layer operates on this observed candidate distribution.


---

2. Split proxy

Definition

The Split proxy measures how much probability mass is fragmented rather than concentrated in a single dominant candidate.

In v0, Split is defined as:

S_t = 1 - max_i P_t(c_i)

where:

p_max(t) = max_i P_t(c_i)

So equivalently:

S_t = 1 - p_max(t)


---

Interpretation of Split

Low Split

If one candidate dominates strongly:

p_max(t) ~ 1

then:

S_t ~ 0

This corresponds to a monolithic or highly consolidated candidate state.

High Split

If the leading candidate does not dominate and probability mass is spread across alternatives, then:

S_t -> 1

in the limit of strong fragmentation.

This corresponds to candidate competition or lack of clear local consolidation.


---

What Split does not measure

Split does not measure:

semantic disagreement

logical inconsistency

correctness

downstream failure by itself


It only measures local fragmentation of probability mass.


---

3. Bifurcation Pressure proxy

Definition

Bifurcation Pressure measures how strongly the candidate hierarchy changes between two consecutive steps.

It is intended to capture structural instability of candidate ordering and probability redistribution.

Version v0 combines two signals:

1. Total Variation shift


2. Rank Swap




---

3.1 Total Variation shift

Given two consecutive observed candidate distributions:

P_t
P_(t+1)

define the union vocabulary:

V = support(P_t) union support(P_(t+1))

Then define Total Variation:

TV_t = 0.5 * sum_x in V |P_t(x) - P_(t+1)(x)|

Interpretation

TV_t measures how much probability mass is redistributed between adjacent steps.

low TV_t = local continuity

high TV_t = strong redistribution



---

3.2 Rank Swap

Define:

top_t = argmax_x P_t(x)
top_(t+1) = argmax_x P_(t+1)(x)

Then Rank Swap is:

RS_t = 1   if top_t != top_(t+1)
RS_t = 0   if top_t = top_(t+1)

Interpretation

This is a binary signal indicating whether the dominant candidate changed across consecutive steps.

It captures discrete hierarchy inversion.


---

3.3 Composite Bifurcation Pressure

Version v0 defines Bifurcation Pressure as:

B_t = 0.7 * TV_t + 0.3 * RS_t

Interpretation

TV_t contributes continuous redistribution signal

RS_t contributes discrete top-rank instability


The weights are provisional. They are not canonical. They are chosen only to create a usable first synthetic composite.


---

What Bifurcation Pressure does not measure

Bifurcation Pressure does not measure:

semantic contradiction

true ambiguity in meaning

correctness of the chosen token

reasoning depth

human-like hesitation


It measures only local reorganization of candidate dominance.


---

4. Sequence-level aggregation

For multi-step traces, proxy values may be aggregated across steps.

Possible sequence-level summaries include:

Split summaries

S_mean = mean_t S_t
S_max  = max_t S_t

Pressure summaries

B_mean = mean_t B_t
B_max  = max_t B_t

These are the summaries used by the v0 synthetic runner.

They are not final sequence metrics.


---

5. Synthetic scenario expectations

The synthetic benchmark is not evidence about real model behavior.

It is only a structural sanity check for the proxy logic.

Scenario 01 — Monolithic

Example profile:

one dominant candidate

negligible alternatives

no adjacent-step competition


Expected behavior:

Split: low
Pressure: zero or not applicable

For the current synthetic example:

S ~ 0.01
B = 0.00


---

Scenario 02 — Bifurcation

Example profile:

two leading candidates near parity

top-rank inversion across adjacent steps

local mass redistribution


Expected behavior:

Split: high
Pressure: high

For the current synthetic example:

step 1 split:


S_1 = 1 - 0.48 = 0.52

step 2 split:


S_2 = 1 - 0.52 = 0.48

split mean:


S_mean = 0.50

For pressure:

TV = 0.5 * (|0.48-0.41| + |0.47-0.52| + |0.03-0.04| + |0.02-0.03|)
   = 0.5 * (0.07 + 0.05 + 0.01 + 0.01)
   = 0.07

Since the top candidate changes from VALID to INVALID:

RS = 1

Therefore:

B = 0.7 * 0.07 + 0.3 * 1 = 0.049 + 0.3 = 0.349

So the expected pressure is:

B ~ 0.349


---

Scenario 03 — Collapse

Example profile:

flat candidate mass

no dominant trajectory

single-step fragmentation only


Expected behavior:

Split: maximal or near-maximal
Pressure: not defined for single-step case

For the current synthetic example:

S = 1 - 0.10 = 0.90

Since only one step exists:

B = N/A


---

6. Why this formalization is useful

This v0 formalization provides three immediate benefits:

1. it makes OMNIAMIND computable without a live backend


2. it tests whether the proxy logic behaves sensibly on controlled synthetic traces


3. it protects the project from premature semantic contamination



The synthetic runner is not a substitute for real backend data.

But it is a valid structural test bench for proxy behavior.


---

7. Known limitations

This proxy layer is intentionally weak in several ways.

Split limitations

S_t = 1 - p_max(t) is simple but incomplete.

It does not distinguish between:

one strong rival

many weak rivals

broad vs narrow dispersion beyond the winner


A later version may incorporate concentration or entropy-like alternatives, but not yet.

Pressure limitations

B_t depends only on adjacent-step redistribution and top-rank swap.

It does not capture:

longer temporal memory

branch persistence

latent geometry

explicit reconvergence structure


General limitation

This entire layer operates only on the observed candidate distribution. It does not access hidden states, branch traces, or semantic structure.

That is acceptable for v0.


---

8. Status

This document defines the first structural proxy layer for OMNIAMIND.

Current status:

non-semantic boundary: preserved

synthetic runner compatibility: yes

real backend dependency: not required for v0 sanity testing

empirical validation on real model traces: not yet performed



---

Minimal conclusion

OMNIAMIND v0 begins with two structural proxies:

Split:
S_t = 1 - max_i P_t(c_i)

Bifurcation Pressure:
B_t = 0.7 * TV_t + 0.3 * RS_t

where:

TV_t measures probability redistribution across adjacent steps

RS_t measures top-rank inversion


These proxies are not final metrics.

They are the first deterministic structural probes for testing candidate fragmentation and candidate instability without semantics.

