# OMNIAMIND (v2.0-alpha)

**Structural Measurement of Split and Bifurcation Pressure in LLM Candidate Traces**

**Author:** Massimiliano Brighindi  
**Project:** MB-X.01

---

## What OMNIAMIND is

OMNIAMIND is a structural measurement layer for pre-output candidate dynamics in large language models.

It does not evaluate final output quality.  
It does not interpret meaning.  
It does not model psychology.  
It does not claim cognition.

Its purpose is narrower:

- measure how candidate probability mass fragments at a given step
- measure how candidate dominance reorganizes across adjacent steps

The current version focuses on two structural proxies:

- **Split**
- **Bifurcation Pressure**

---

## Core boundary

OMNIAMIND is strictly **non-semantic**.

It treats the model as an evolving candidate distribution, not as a linguistic agent.

It does not measure:

- truth
- correctness
- semantic similarity
- reasoning quality
- human-like uncertainty

It measures only the geometry of candidate probability structure.

---

## Architectural position

OMNIAMIND belongs to the MB-X.01 ecosystem and sits upstream of OMNIA.

```text
Dual-Echo -> OMNIAMIND -> OMNIA -> OMNIA-LIMIT

Within this chain:

OMNIAMIND measures pre-output structural dynamics

OMNIA measures post-hoc output stability under controlled transformations

OMNIA-LIMIT defines structural stop / saturation conditions



---

Ecosystem position

OMNIAMIND is an upstream analytical extension of the OMNIA ecosystem.

It does not replace OMNIA.

OMNIAMIND measures pre-output structural dynamics

OMNIA remains the canonical post-hoc structural measurement engine

OMNIA-LIMIT remains the downstream structural stop / saturation layer


Canonical ecosystem map:
https://github.com/Tuttotorna/lon-mirror/blob/main/ECOSYSTEM.md


---

Current scope

The current repository state is split into two branches:

Branch	Status	Note

Analytical Branch	OPERATIONAL	Proxy definitions, synthetic traces, deterministic runner, and internal comparison logic are available.
Backend Branch	UNRESOLVED	Real capture through OpenAI or local runtime is defined but not executed.


Current milestone:

Repository-level clarity and synthetic proxy validation


---

Current proxy layer (v0)

1. Split

Split measures local fragmentation of candidate probability mass at a single generation step.

Version v0:

S_t = 1 - max_i P_t(c_i)

Interpretation:

low Split -> one dominant candidate

high Split -> fragmented candidate mass


2. Bifurcation Pressure

Bifurcation Pressure measures local instability between adjacent candidate states.

Version v0 combines:

Total Variation shift

Rank Swap


B_t = 0.7 * TV_t + 0.3 * RS_t

where:

TV_t measures redistribution of probability mass across adjacent steps

RS_t detects whether the top-ranked candidate changed


This is a structural proxy only.
It is not a semantic or cognitive claim.


---

Repository structure

README.md
STATUS.md
MANIFEST.md
result_schema.py
synthetic_proxy_runner.py
report_generator.py
proxy_variants.py
proxy_variant_runner.py
variant_report_generator.py
docs/
  PROXY_FORMALIZATION_v0.md
  CORE_PHENOMENA.md
  MEASUREMENT_SKETCH.md
  ACCESS_LEVELS.md
  FIRST_EXPERIMENT_PATH.md
  FIRST_PROXIES.md
  backend_assessments/
    OPENAI_CHECK.md
    OPENAI_RESULT.md
    OPENAI_EMPIRICAL_CHECK.md
    LOCAL_CHECK.md
    LOCAL_RUNTIME_STATUS.md
    LLAMA_CPP_CHECK.md
    LLAMA_CPP_RESULT.md
data/
  synthetic_proxy_traces_v0.json
  synthetic_proxy_results_v0.json
  proxy_variant_comparison_v0_1.json
reports/
  synthetic_proxy_report_v0.md
  proxy_variant_report_v0_1.md

Main files:

docs/PROXY_FORMALIZATION_v0.md — mathematical definition of the v0 proxy layer

data/synthetic_proxy_traces_v0.json — synthetic benchmark traces

synthetic_proxy_runner.py — deterministic proxy runner over synthetic traces

proxy_variants.py — structural proxy variants (v0 and v0.1)

proxy_variant_runner.py — internal comparison runner for proxy variants

docs/backend_assessments/ — backend readiness and assessment documents



---

Quick start (synthetic branch)

The repository can already be exercised without any active backend.

1. Run the baseline proxy engine

python synthetic_proxy_runner.py

2. Generate the baseline Markdown report

python report_generator.py

3. Run the proxy variant comparison

python proxy_variant_runner.py

4. Generate the variant comparison report

python variant_report_generator.py

This branch processes synthetic traces and writes proxy results for the predefined scenarios.

The purpose of this step is not empirical model validation.
It is structural sanity checking of the proxy logic.


---

Synthetic scenarios currently included

The current synthetic dataset contains three minimal controlled cases.

scenario_01_monolithic

Single dominant candidate, negligible competition.

Expected behavior:

low Split

zero Pressure


scenario_02_bifurcation

Two near-parity candidates with top-rank inversion across adjacent steps.

Expected behavior:

high Split

high Pressure


scenario_03_collapse

Flat candidate mass with no dominant continuation.

Expected behavior:

maximal or near-maximal Split

Pressure not applicable in the single-step case



---

Declared limits

OMNIAMIND v0 has explicit limits.

1. No semantics

The system does not know whether two candidates are synonymous, contradictory, or equivalent in meaning.

2. Observed-space only

The current proxy layer operates only on exposed candidate distributions.

It does not use:

hidden states

attention maps

branch traces

latent geometry


3. Synthetic branch is not empirical validation

Synthetic traces are only a structural test bench.

They do not prove that the proxies will behave identically on real model traces.

4. Backend branch remains unresolved

No OpenAI or local llama.cpp empirical capture has yet been completed inside the current project state.


---

Why this repository exists in its current form

The project reached an infrastructure block on real backend execution.

Instead of fabricating empirical claims, the repository was stabilized around what could be completed with full internal consistency:

proxy definition

mathematical formalization

synthetic validation traces

deterministic runner

internal proxy comparison

backend assessment protocols


This means the repository is analytically stable even though empirical backend validation is still pending.


---

Next steps

The next valid milestones are:

1. execute a real backend capture


2. classify the first backend as ACCEPT, PROVISIONAL, or REJECT


3. run the first OMNIAMIND proxy computation on real traces


4. compare synthetic behavior against observed backend traces


5. revise proxy coefficients only if empirical behavior justifies it



Until then, the correct project state is:

Analytical branch: operational
Backend branch: unresolved


---

Minimal conclusion

OMNIAMIND is a non-semantic structural measurement layer for pre-output candidate dynamics.

Its current v0 / v0.1 state is operational in the analytical branch through:

formal proxy definitions

synthetic traces

deterministic execution stubs

internal proxy comparison


The backend branch is prepared but not yet empirically resolved.

This repository therefore represents a structurally complete analytical core awaiting real candidate-trace input.