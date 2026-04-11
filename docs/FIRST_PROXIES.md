# OMNIAMIND — First Proxies

**Author:** Massimiliano Brighindi  
**Project:** MB-X.01  
**Target access level:** Level 1 — Token / Candidate Exposure

---

## Purpose

This document defines the first provisional proxy quantities for OMNIAMIND.

These proxies are not validated metrics.
They are not final formulas.
They are not universal.

They are the first computable approximations for two pre-output phenomena:

- **Split**
- **Bifurcation Pressure**

The goal is narrow:

- define the minimum observable inputs
- define simple proxy quantities
- define what would count as positive, negative, or inconclusive signal behavior

This document marks the transition from conceptual definition to first experimental measurement.

---

## Position in the ecosystem

```text
Dual-Echo -> OMNIAMIND -> OMNIA -> OMNIA-LIMIT -> external decision layer

Within this chain:

OMNIAMIND measures pre-output structural dynamics

OMNIA provides post-hoc structural measurement on emitted output

OMNIA-LIMIT defines structural stop / saturation conditions


The proxies below must remain compatible with this boundary.


---

Scope

This first proxy layer is intentionally minimal.

It does not try to measure:

semantics

correctness

truth

reasoning quality

latent geometry

full branch reconstruction


It only tries to detect whether candidate competition at Level 1 contains usable upstream structural signal.


---

Minimum required input

The first proxy experiment assumes Level 1 access.

Minimum required observations per generation step t:

C_t: a set of top-k candidate items exposed at step t

P_t(c_i): probability-like or score-like quantity associated with each candidate c_i in C_t


Possible examples of candidate items:

top-k next tokens

ranked continuation fragments

beam candidates if available


This document does not force a single substrate. However, every experiment must state clearly which candidate type is being used.


---

Input constraints

The following constraints apply immediately:

1. Candidate identity must be explicit

The experiment must define what a candidate is at each step.

A token is not the same object as a multi-token continuation. A ranked item is not the same as a branch trace.

No proxy result is interpretable without this declaration.

2. Score type must be explicit

The experiment must declare whether P_t(c_i) refers to:

probability

logprob

normalized score

rank-derived surrogate


No hidden conversion should be assumed.

3. Top-k choice must be explicit

The value of k must be reported. Proxy behavior may depend strongly on this value.


---

Proxy A — Split

Target

Approximate the emergence of meaningful candidate divergence at a given generation step.

Intuition

A split signal should rise when the candidate set can no longer be described as one strongly dominant, weakly perturbed continuation.

The signal should reflect both:

reduced dominance concentration

increased candidate separation


A flat top-k list of near-identical alternatives should not count as a strong split.


---

A1. Split concentration proxy

Purpose

Estimate whether single-path dominance is weakening.

Sketch

Let:

p_max(t) = highest candidate weight at step t

mass_topm(t) = total mass of the top m candidates, with m <= k


A simple concentration-based split proxy can be defined as:

S_conc(t) = 1 - p_max(t)

or, slightly less local:

S_mass(t) = 1 - mass_topm(t)

Interpretation

Higher values suggest weaker dominance and therefore greater possibility of branching competition.

Limitation

This alone does not prove meaningful split. Low concentration may still occur among very similar candidates.


---

A2. Split dispersion proxy

Purpose

Estimate whether leading candidates are structurally separated from one another.

Sketch

Assume a distance function d(c_i, c_j) over the candidate items exposed at step t.

Then a simple dispersion proxy can be defined as weighted average pairwise distance:

S_disp(t) = weighted_mean of d(c_i, c_j) over selected pairs in C_t

The weighting rule may depend on candidate scores, but must be reported explicitly.

Interpretation

Higher values suggest that candidate competition is not only flatter, but also more structurally separated.

Limitation

This depends heavily on the candidate representation and chosen distance function. String distance, token distance, and continuation distance are not equivalent.


---

A3. Composite split proxy

Purpose

Combine weakening dominance and candidate separation into one provisional signal.

Sketch

A first composite form may be:

S_proxy(t) = f(S_conc(t), S_disp(t))

where f is a simple monotone combination such as weighted sum, normalized product, or other explicitly declared rule.

Interpretation

A stronger split signal should require both:

reduced dominance concentration

non-trivial candidate dispersion


Limitation

The combination rule is provisional and should not be treated as canonical.


---

Proxy B — Bifurcation Pressure

Target

Approximate the instability that pushes the system toward incompatible continuations before or during visible split.

Intuition

Bifurcation pressure is not split itself.

It is the local instability of candidate competition: the tendency of dominant paths to become unstable, exchange rank, or fluctuate in relative strength across adjacent steps.


---

B1. Dominance volatility proxy

Purpose

Estimate how unstable the candidate hierarchy is across adjacent generation steps.

Sketch

Let the highest-weight candidates at steps t-1 and t be compared through score change or ordering change.

A simple volatility proxy may track:

B_vol(t) = magnitude of change in leading candidate weights or ranks from t-1 to t

Possible implementations include:

change in top-1 score

change in top-2 gap

rank instability among top candidates


Interpretation

Higher values suggest stronger instability in local dominance structure.

Limitation

Volatility may also reflect benign local variation. It should not be read as pressure unless it persists or aligns with other signals.


---

B2. Candidate mass redistribution proxy

Purpose

Estimate how strongly probability-like mass is moving across leading candidates.

Sketch

A simple redistribution proxy may track the stepwise change in the candidate score distribution:

B_redist(t) = distribution_change(P_t, P_(t-1))

The exact distribution-change function must be declared explicitly. Possible choices may include simple norm differences or divergence-like quantities.

Interpretation

Higher values suggest stronger local rearrangement of candidate competition.

Limitation

This still does not show incompatibility by itself. It only shows instability in candidate mass allocation.


---

B3. Top-rank swap proxy

Purpose

Detect local competition instability among leading candidates.

Sketch

A simple binary or counted signal may record whether the ordering of top candidates changes across adjacent steps:

B_swap(t) = indicator or count of rank-order changes among top candidates from t-1 to t

Interpretation

Repeated swap events suggest unstable candidate dominance and possible approach to branching.

Limitation

A single swap event is weak evidence. This proxy is more useful as a repeated pattern than as a one-step fact.


---

B4. Composite bifurcation-pressure proxy

Purpose

Combine local dominance instability and mass redistribution into one provisional pressure signal.

Sketch

A first composite form may be:

B_proxy(t) = g(B_vol(t), B_redist(t), B_swap(t))

where g is an explicitly declared monotone combination rule.

Interpretation

Pressure should rise when:

dominant candidates become unstable

mass shifts strongly across leading alternatives

ranking changes persist across adjacent steps


Limitation

This is still a proxy. It does not yet prove downstream incompatibility.


---

Stepwise vs sequence-level aggregation

The proxies above are defined locally at generation step t.

A full experiment will likely need sequence-level summaries.

Possible aggregations include:

max value across steps

mean value across steps

persistence over threshold

area under signal curve

onset time of first elevated signal


The aggregation rule must always be reported.

No single aggregation is assumed here.


---

Positive, negative, and inconclusive outcomes

Positive

A result is provisionally positive if one or both proxy families show directional separation between downstream cleaner cases and downstream weaker cases.

Examples:

S_proxy tends to be higher in cases later measured as more fragile by OMNIA-like post-hoc analysis

B_proxy rises earlier or more persistently in those same cases

the separation remains visible across repeated runs or across small parameter changes


This would count as initial evidence of useful upstream structural visibility.


---

Negative

A result is negative if the proxy families fail to provide useful discrimination.

Examples:

proxy values do not differ meaningfully across compared conditions

variation is dominated by noise or arbitrary candidate artifacts

repeated runs erase the apparent effect

no added signal appears beyond what final-output analysis already shows


A negative result is acceptable. It would constrain OMNIAMIND rather than invalidate the entire framework.


---

Inconclusive

A result is inconclusive if signals appear but remain too unstable or too implementation-dependent.

Examples:

results change drastically with k

results depend almost entirely on one distance function choice

one backend shows separation while another collapses

apparent signal disappears under mild repetition


Inconclusive means the proxy layer is not yet strong enough, not that the concept is either confirmed or refuted.


---

Reporting rule

Any first-proxy experiment must report at least:

1. access level used


2. candidate type used


3. score type used


4. value of k


5. distance function used, if any


6. aggregation rule used


7. comparison groups used


8. whether the outcome is positive, negative, or inconclusive



Without all eight items, the result is underspecified.


---

Non-goals

This document does not define:

final OMNIAMIND metrics

universal formulas

architecture-independent guarantees

production thresholds

abort policy

latent-state measurements


Those belong to later stages only if these first proxies show repeatable directional utility.


---

Minimal conclusion

The first operational OMNIAMIND proxy layer should begin with two targets:

Split

Bifurcation Pressure


At Level 1, these can be approximated through:

candidate concentration

candidate dispersion

dominance volatility

score redistribution

rank instability


These are not final metrics.

They are the first computable probes for testing whether pre-output candidate exposure contains usable upstream structural signal.



 successivo corretto, senza saltare troppo avanti:

