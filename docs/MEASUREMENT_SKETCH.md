# OMNIAMIND — Measurement Sketch

**Author:** Massimiliano Brighindi  
**Project:** MB-X.01

---

## Purpose

This document sketches possible measurement strategies for the six OMNIAMIND core phenomena.

It does not define validated formulas.
It does not claim universal metrics.
It does not assume a single model architecture.

Its purpose is narrower:

- to map each phenomenon to measurable candidates
- to identify what kind of signal would be needed
- to separate plausible measurement directions from premature formalization

This is a measurement sketch, not a final specification.

---

## Position in the ecosystem

```text
Dual-Echo -> OMNIAMIND -> OMNIA -> OMNIA-LIMIT -> external decision layer

Within this chain:

OMNIAMIND measures pre-output structural dynamics

OMNIA measures post-hoc output stability under controlled transformations

OMNIA-LIMIT certifies structural saturation / stop conditions


OMNIAMIND is upstream. It does not evaluate final correctness. It does not interpret semantics. It does not decide.


---

Scope of measurement

OMNIAMIND assumes access to a sequence of candidate pre-output structural states.

These states may be approximated through different substrates, depending on what the system exposes.

Possible substrates may include:

candidate token distributions

ranked continuation sets

latent state snapshots

internal branch approximations

intermediate structural traces

externally reconstructed candidate paths


This document remains substrate-agnostic.

The only hard requirement is that some notion of candidate-state comparison exists across steps.


---

General measurement principles

Any future OMNIAMIND metric should satisfy the following principles:

1. Structural only

Measurement must not require semantic interpretation.

2. Architecture tolerance

Measurement should remain meaningful across different model families or approximations.

3. Comparative, not absolute

Whenever possible, signals should be based on relative change, divergence, persistence, or recovery patterns rather than absolute values.

4. Local falsifiability

A metric should make it possible to observe when the target phenomenon is absent, weak, rising, or collapsing.

5. No premature universality

A useful sketch is better than a false universal law.


---

Shared notation (informal)

This document uses informal placeholders only.

Let:

S_t = candidate structural state at step t

C_t = set of candidate paths or candidate-state approximations at step t

d(a, b) = structural distance between two candidate states or paths

sim(a, b) = structural similarity between two candidate states or paths

P_t = distribution or weighting over candidate paths at step t


These are placeholders, not finalized definitions.

They only indicate the type of quantity that may later be used.


---

1. Split

Target

Measure the emergence of multiple candidate trajectories from a previously more unified path.

Measurement intuition

A split becomes measurable when the candidate structure can no longer be described as a single dominant coherent path.

The signal is not just multiplicity. The signal is meaningful branch separation.

Possible measurement directions

A. Branch count above separation threshold

Estimate how many candidate branches remain distinguishable once a minimum separation criterion is applied.

Sketch:

split_count(t) = number of branches in C_t such that pairwise distance exceeds epsilon_split

B. Mean pairwise branch distance

Measure the average separation among top candidate branches.

Sketch:

split_dispersion(t) = mean d(c_i, c_j) over selected pairs in C_t

C. Branch dominance flattening

If one path stops dominating and multiple paths gain comparable weight, split likelihood rises.

Sketch:

split_balance(t) = concentration drop in P_t

Minimal signal interpretation

Split should rise when:

branch count rises

branch separation rises

branch dominance becomes less concentrated


Caution

High candidate count alone is not sufficient. Many near-identical candidates do not imply a meaningful split.


---

2. Micro-drift

Target

Measure cumulative sub-threshold deviation along a candidate trajectory before visible downstream instability appears.

Measurement intuition

Micro-drift is weak but persistent deviation. It should not depend on a single abrupt break.

Possible measurement directions

A. Incremental path deviation

Measure step-to-step deviation of the dominant or tracked candidate path relative to its recent trajectory.

Sketch:

drift_step(t) = d(S_t, expected_local_path(t))

B. Cumulative drift integral

Accumulate small deviations over a moving window.

Sketch:

micro_drift(T) = sum of drift_step(t) over t in window T

C. Curvature of trajectory

Estimate whether the path is bending away from prior local coherence even without full branching.

Sketch:

trajectory_curvature(t) = change in direction of structural evolution across recent steps

Minimal signal interpretation

Micro-drift should rise when:

local deviation persists

cumulative deviation grows

path curvature departs from prior coherence


Caution

Micro-drift must remain distinct from noise spikes. The target is accumulation, not isolated fluctuation.


---

3. Echo Persistence

Target

Measure how strongly earlier candidate paths continue to influence later candidate states after apparent displacement.

Measurement intuition

A path may appear replaced while still leaving structural residue in later states.

Echo persistence tries to quantify that residue.

Possible measurement directions

A. Lagged similarity

Measure similarity between an earlier branch and later candidate states across a lag window.

Sketch:

echo_lag(i, t) = sim(branch_i at time tau, candidate state at time t)

for tau < t.

B. Residual branch contribution

Track whether previously weakened or displaced branches still contribute to later candidate composition.

Sketch:

echo_residual(t) = retained influence of prior branches on current candidate mixture

C. Persistence decay profile

Model how quickly earlier branch signatures disappear.

Sketch:

echo_half_life = time needed for prior branch influence to decay below threshold

Minimal signal interpretation

Echo persistence should rise when:

earlier paths remain detectable later

branch influence decays slowly

later states retain structure from displaced paths


Caution

Echo persistence is not recurrence in a semantic sense. It is only residual structural carryover.


---

4. Reconvergence

Target

Measure the return of previously diverging branches toward structural realignment before final emission.

Measurement intuition

Reconvergence is not just lower dispersion. It is directional recovery from prior divergence.

Possible measurement directions

A. Pairwise distance contraction

Measure whether branch distances shrink over time after a split phase.

Sketch:

reconvergence_rate(t) = negative slope of mean pairwise branch distance

B. Dominance re-consolidation

Measure whether branch weighting returns toward a more concentrated state after a dispersion phase.

Sketch:

reconvergence_concentration(t) = concentration recovery in P_t after branching

C. Terminal coherence restoration

Estimate how much of the earlier branch fragmentation is resolved near emission.

Sketch:

terminal_recovery = reduction of split-related dispersion near final pre-output steps

Minimal signal interpretation

Reconvergence should rise when:

branch separation decreases

branch competition stabilizes

terminal pre-output structure becomes more unified


Caution

Reconvergence is not proof of truth or correctness. It only indicates restored structural alignment among candidates.


---

5. Bifurcation Pressure

Target

Measure the degree of structural pressure pushing the system toward incompatible downstream paths, even before a stable split is fully visible.

Measurement intuition

Bifurcation pressure is an instability precursor. It may appear before branch count is clearly elevated.

Possible measurement directions

A. Candidate dominance volatility

Measure how unstable the relative ordering or weighting of leading candidate paths becomes.

Sketch:

pressure_volatility(t) = short-horizon instability in top-path weights or rankings

B. Local sensitivity to perturbation

Test whether small changes in the current state or candidate approximation produce large downstream path differences.

Sketch:

pressure_sensitivity(t) = downstream path divergence induced by small local perturbation

C. Incompatibility growth

Measure whether top candidate paths are becoming more mutually exclusive over time.

Sketch:

pressure_incompatibility(t) = growth rate of incompatibility among top candidate continuations

Minimal signal interpretation

Bifurcation pressure should rise when:

top candidate ordering becomes unstable

small perturbations yield large branch changes

incompatibility among plausible continuations grows


Caution

Pressure is not identical to split. Pressure may rise before clean branching is visible.


---

6. Collapse Threshold

Target

Estimate proximity to the region where reconvergence becomes structurally negligible or impossible.

Measurement intuition

Collapse threshold is not a single universal point. It is a recoverability boundary relative to the observed dynamics.

Possible measurement directions

A. Reconvergence decay

Track whether the probability or tendency of branch recovery falls below a recoverability range.

Sketch:

collapse_proximity(t) = inverse of expected reconvergence capacity from current state

B. Persistent separation beyond recovery window

Measure whether branch distances remain high for too long relative to the system's observed recovery behavior.

Sketch:

collapse_persistence(t) = duration of unresolved branch separation beyond recovery horizon

C. Irreversibility indicators

Estimate whether the current branch configuration has entered a regime with negligible path contraction.

Sketch:

collapse_irreversibility(t) = failure of branch contraction under continued evolution

Minimal signal interpretation

Collapse-threshold proximity should rise when:

reconvergence tendency falls

unresolved separation persists

recovery signals decay toward negligible levels


Caution

This signal should remain conditional. It can justify a possible future gating mechanism, but it is not itself a decision.


---

Cross-phenomenon relations

The six targets are structurally related.

A provisional measurement chain may look like this:

1. local coherence weakens through micro-drift


2. candidate competition raises bifurcation pressure


3. split becomes measurable


4. prior paths remain active through echo persistence


5. reconvergence either restores structural alignment


6. or collapse-threshold proximity rises toward non-recoverable divergence



This is not a mandatory sequence. Real systems may:

skip stages

loop between stages

show overlap between drift and pressure

split without meaningful reconvergence

reconverge after long persistence


The purpose of the chain is diagnostic orientation only.


---

Data access levels

Different implementations may have different visibility.

Possible levels include:

Level 0 — external approximation only

Only output-facing candidate approximations are available.

Level 1 — token-level candidate exposure

Top candidate continuations or probability slices are available.

Level 2 — latent-state snapshots

Intermediate internal state representations are accessible.

Level 3 — branch-trace instrumentation

Approximate candidate trajectories can be tracked explicitly.

OMNIAMIND does not require all levels. But any implementation must state clearly which access level it assumes.


---

Minimal implementation strategy

A reasonable first implementation path should be modest.

Not:

full universal metric suite

model-agnostic perfection

production gating from day one


But rather:

1. choose one access level


2. define one or two provisional quantities per phenomenon


3. test whether those quantities vary in the expected direction


4. check whether they provide additional information beyond post-hoc OMNIA alone



This is the only sane order.


---

Boundary conditions

This document must not be reinterpreted as:

a theory of thought

a consciousness detector

a semantic parser

a general truth engine

a decision policy


It is only a sketch of possible measurements for pre-output structural dynamics.


---

Status

This document is exploratory.

It defines:

possible measurement directions

placeholder quantities

substrate-agnostic implementation sketches


It does not define:

validated metrics

official thresholds

benchmark claims

deployment rules



---

Minimal conclusion

OMNIAMIND v2.0 begins here:

not with final formulas, but with explicit measurement sketches for the six core pre-output phenomena:

split

micro-drift

echo persistence

reconvergence

bifurcation pressure

collapse threshold


The next valid step is not branding. It is testing whether any of these sketches can be turned into measurable, falsifiable signals on a chosen access layer.

Questo file è giusto perché fa il salto corretto:
da **definizioni** a **forme di misura possibili**, senza fingere di avere già una teoria chiusa.

Dopo questo, il passo migliore è uno solo:

**Nome file**
```text
docs/ACCESS_LEVELS.md

