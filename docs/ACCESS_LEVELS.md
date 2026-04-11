# OMNIAMIND — Access Levels

**Author:** Massimiliano Brighindi  
**Project:** MB-X.01

---

## Purpose

This document defines the access levels under which OMNIAMIND may operate.

OMNIAMIND is a pre-output structural diagnostics layer.
Its measurement possibilities depend directly on what kind of internal or quasi-internal data can be observed.

The purpose of this document is to prevent category error.

It does not define metrics.
It does not define benchmark results.
It does not assume that all implementations have the same visibility.

It defines what kinds of pre-output structural signals are realistically accessible under different observation regimes.

---

## Position in the ecosystem

```text
Dual-Echo -> OMNIAMIND -> OMNIA -> OMNIA-LIMIT -> external decision layer

Within this chain:

OMNIAMIND measures pre-output structural dynamics

OMNIA measures post-hoc output stability under controlled transformations

OMNIA-LIMIT certifies structural saturation / stop conditions


OMNIAMIND is upstream relative to emitted output. Its feasibility depends on access conditions.


---

Why access levels matter

A measurement layer is only meaningful if its required signals are actually obtainable.

Without explicit access constraints, OMNIAMIND risks becoming:

overfitted to idealized instrumentation

falsely universal

impossible to reproduce

detached from real deployment conditions


Access levels provide the realism boundary.

They specify:

what can be observed

what can only be approximated

what cannot be claimed under a given regime



---

General principle

The deeper the access, the richer the possible pre-output diagnostics.

But deeper access is not always available.

Therefore:

lower access levels require stronger approximation discipline

higher access levels allow stronger structural claims

every OMNIAMIND implementation must declare its assumed access level explicitly


No result is interpretable without this declaration.


---

Access Level 0 — External Approximation Only

Definition

At Level 0, OMNIAMIND has no direct access to internal candidate states.

The only available data come from repeated external interaction with the system.

Possible observable artifacts include:

multiple sampled outputs

repeated completions from similar prompts

perturbation-response behavior

ranked outputs if externally exposed

completion variability across runs


What is accessible

Level 0 can observe:

output variability across repeated runs

divergence patterns under repeated prompting

delayed instability signatures inferred from output ensembles

approximate branch competition reconstructed from externally visible alternatives


What is not accessible

Level 0 cannot directly observe:

internal latent states

token-by-token candidate distributions unless exposed

internal branching dynamics

hidden candidate persistence

true pre-output trajectory geometry


What OMNIAMIND can plausibly do at Level 0

At this level, OMNIAMIND can only build proxy diagnostics.

Possible targets:

approximate split via output ensemble dispersion

approximate micro-drift via sensitivity to small prompt perturbations

approximate reconvergence via cross-run clustering near terminal outputs

approximate collapse proximity via repeated unrecoverable divergence patterns


Main limitation

Level 0 is inference from aftermath, not direct observation of pre-output structure.

It is the weakest level and must remain explicitly approximate.

Strength

It is the most portable level.

It can be used even with opaque black-box systems.


---

Access Level 1 — Token / Candidate Exposure

Definition

At Level 1, OMNIAMIND can observe partial pre-output candidate information during generation.

This may include:

token probability distributions

top-k candidate tokens per step

logprobs

ranked continuation slices

beam candidates if exposed


What is accessible

Level 1 can observe:

candidate competition at each generation step

local dominance changes

branch instability in top-token or top-path rankings

partial pre-output dispersion

local reconsolidation after temporary competition


What is not accessible

Level 1 still cannot fully observe:

deeper latent geometry unless separately exposed

full hidden-state dynamics

internal candidate structure below the exposed token layer

branch traces not represented in the exposed distributions


What OMNIAMIND can plausibly do at Level 1

Possible targets:

split via top-k dispersion and candidate separation

bifurcation pressure via volatility in token/path ranking

micro-drift via cumulative shift in local candidate distributions

reconvergence via return to dominant concentration after dispersion

collapse-threshold proximity via persistent instability in candidate concentration


Main limitation

Level 1 exposes only the surface of pre-output dynamics.

It is stronger than Level 0, but still incomplete.

Strength

This is likely the first practically useful level for an experimental OMNIAMIND implementation.

It provides real pre-output information without requiring privileged model internals.


---

Access Level 2 — Latent-State Snapshots

Definition

At Level 2, OMNIAMIND can access internal state representations during generation.

Possible examples include:

hidden-state vectors

residual stream snapshots

internal activations at selected layers

compressed latent summaries

stepwise state embeddings


What is accessible

Level 2 can observe:

trajectory motion in internal representation space

local coherence and deformation of hidden paths

branch separation beyond token-level competition

persistence of internal structural traces across steps

richer reconvergence patterns not visible in emitted token rankings


What is not accessible

Level 2 may still not expose:

explicit native branching if the architecture does not represent it directly

a guaranteed canonical geometry across models

full interpretability of state dimensions

universal comparability across architectures without careful normalization


What OMNIAMIND can plausibly do at Level 2

Possible targets:

split via latent branch separation

micro-drift via cumulative deviation in latent trajectory

echo persistence via lagged similarity across latent states

reconvergence via contraction of latent branch distance

bifurcation pressure via local instability and sensitivity in latent evolution

collapse-threshold proximity via sustained non-recoverable separation in state space


Main limitation

Latent access increases measurement power but also increases comparability problems.

Not all latent spaces are directly meaningful without normalization, alignment, or local geometry assumptions.

Strength

This is the strongest general-purpose level for true pre-output structural diagnostics without assuming architecture-specific native branch tracing.


---

Access Level 3 — Explicit Branch Trace Instrumentation

Definition

At Level 3, OMNIAMIND can observe explicit candidate trajectories as structured branches during generation.

This may happen when the system exposes or can reconstruct:

beam search branches

multi-path candidate traces

branch scores over time

structured rollout trees

explicit branch competition histories


What is accessible

Level 3 can observe:

branch birth

branch persistence

branch merger or reconvergence

branch competition over time

recoverability vs irreversibility of divergence

path-dependent pre-output structure


What is not accessible

Even at Level 3, OMNIAMIND does not automatically gain:

semantics

truth

correctness

decision authority


Explicit branching is still structural data only.

What OMNIAMIND can plausibly do at Level 3

Possible targets:

split via explicit branch emergence count

micro-drift via path deformation within each branch

echo persistence via measurable influence of prior branches on later paths

reconvergence via explicit branch contraction or merger

bifurcation pressure via instability in relative branch strength

collapse-threshold proximity via explicit failure of reconvergence over tracked branch histories


Main limitation

Level 3 often depends on architecture-specific instrumentation or decoding setup.

It may be difficult to generalize across systems.

Strength

This is the cleanest level for direct operationalization of OMNIAMIND core phenomena.

It minimizes approximation and maximizes structural observability.


---

Comparative summary

Level 0

Visibility: external outputs only

Signal type: indirect proxy

Portability: very high

Precision: low

Best use: black-box exploratory diagnostics


Level 1

Visibility: token / top-k candidate exposure

Signal type: partial pre-output

Portability: moderate to high

Precision: moderate

Best use: first practical experimental OMNIAMIND layer


Level 2

Visibility: latent-state snapshots

Signal type: internal trajectory geometry

Portability: moderate

Precision: high if geometry is handled carefully

Best use: stronger research-grade structural diagnostics


Level 3

Visibility: explicit branch traces

Signal type: direct branch dynamics

Portability: low to moderate

Precision: highest

Best use: architecture-aware direct branch analysis



---

Consequences for claims

OMNIAMIND claims must scale with access level.

At Level 0

Claims must remain explicitly proxy-based.

At Level 1

Claims may describe local pre-output competition, but not deep internal geometry.

At Level 2

Claims may describe internal trajectory structure, but only relative to the chosen state representation and normalization regime.

At Level 3

Claims may describe explicit branch dynamics, but still only as structural diagnostics, not semantic or cognitive facts.


---

Minimum reporting rule

Any OMNIAMIND experiment, benchmark, or implementation should report at least:

1. assumed access level


2. observed substrate


3. comparison function used


4. approximation method if direct access is absent


5. limitations induced by that level



Without this, results are underspecified.


---

Design implications

The correct development order is not to start from the richest level by fantasy.

The correct order is:

1. identify the lowest realistically accessible level


2. define provisional quantities compatible with that level


3. test whether those quantities vary in the expected direction


4. only then escalate to deeper access if needed



This prevents OMNIAMIND from becoming a theory that depends on unavailable data.


---

Boundary conditions

Access depth does not change OMNIAMIND's role.

At every level, OMNIAMIND remains:

non-semantic

non-psychological

non-decisional

non-anthropomorphic


More access improves visibility. It does not license stronger metaphysical claims.


---

Status

This document defines the operational realism boundary for OMNIAMIND.

It does not define:

final metrics

benchmark claims

universal observability

production instrumentation standards


It defines only the access regimes under which OMNIAMIND may attempt measurement.


---

Minimal conclusion

OMNIAMIND is only as real as the access layer that supports it.

The four access levels are:

Level 0 — external approximation only

Level 1 — token / candidate exposure

Level 2 — latent-state snapshots

Level 3 — explicit branch trace instrumentation


Every future OMNIAMIND result must declare which level it assumes.

Without that declaration, pre-output structural claims remain underspecified.

