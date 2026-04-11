Nome file

MANIFEST.md

Contenuto completo

# OMNIAMIND — Offline Analytical Package Manifest

**Author:** Massimiliano Brighindi  
**Project:** MB-X.01  
**Package state:** Frozen offline analytical package v0 / v0.1

---

## Purpose

This file defines the exact scope of the current OMNIAMIND offline package.

It exists to freeze the analytical branch in a clear and minimal form.

This package is:

- runnable
- non-semantic
- synthetic
- structurally coherent
- independent from external APIs or local model runtime execution

This package is **not** an empirical backend validation artifact.

---

## Package identity

OMNIAMIND in this frozen state is:

```text
a non-semantic structural proxy engine for synthetic candidate traces

It currently supports:

baseline proxy execution (v0)

internal proxy comparison (v0 vs v0.1)

structured JSON outputs

Markdown report generation



---

Frozen scope

The current frozen package includes only the analytical branch.

It does not include resolved backend execution.

Included

formal proxy definitions

synthetic trace dataset

deterministic proxy runners

structured result schema

Markdown report generators

status and boundary documentation


Excluded

OpenAI empirical capture

llama.cpp empirical capture

real model-trace ingestion

semantic similarity layers

latent-state analysis

production deployment logic



---

Canonical files in the frozen package

Root files

README.md

STATUS.md

MANIFEST.md

result_schema.py

synthetic_proxy_runner.py

report_generator.py

proxy_variants.py

proxy_variant_runner.py

variant_report_generator.py


Documentation

docs/PROXY_FORMALIZATION_v0.md

docs/CORE_PHENOMENA.md

docs/MEASUREMENT_SKETCH.md

docs/ACCESS_LEVELS.md

docs/FIRST_EXPERIMENT_PATH.md

docs/FIRST_PROXIES.md


Backend assessment documents

docs/backend_assessments/OPENAI_CHECK.md

docs/backend_assessments/OPENAI_RESULT.md

docs/backend_assessments/OPENAI_EMPIRICAL_CHECK.md

docs/backend_assessments/LOCAL_CHECK.md

docs/backend_assessments/LOCAL_RUNTIME_STATUS.md

docs/backend_assessments/LLAMA_CPP_CHECK.md

docs/backend_assessments/LLAMA_CPP_RESULT.md


Data

data/synthetic_proxy_traces_v0.json


Generated outputs

data/synthetic_proxy_results_v0.json

data/proxy_variant_comparison_v0_1.json

reports/synthetic_proxy_report_v0.md

reports/proxy_variant_report_v0_1.md



---

Minimal entrypoints

The minimum runnable entrypoints are:

Baseline proxy execution

python synthetic_proxy_runner.py

Baseline report generation

python report_generator.py

Variant comparison execution

python proxy_variant_runner.py

Variant comparison report generation

python variant_report_generator.py


---

Expected generated artifacts

Running the package should produce or refresh these artifacts:

data/synthetic_proxy_results_v0.json
data/proxy_variant_comparison_v0_1.json
reports/synthetic_proxy_report_v0.md
reports/proxy_variant_report_v0_1.md


---

Current proxy coverage

Baseline v0

Split v0

Bifurcation Pressure v0


Comparison variant v0.1

Split v0.1

Bifurcation Pressure v0.1


The purpose of v0.1 is internal comparison only. It is not a declared replacement for v0.


---

Boundary conditions

This frozen package must remain strictly non-semantic.

It must not be described as:

a truth engine

a reasoning evaluator

a semantic classifier

a consciousness detector

a psychological model


It is only a structural proxy system operating on candidate probability traces.


---

Declared limits

1. Synthetic-only validation

All current runs operate on synthetic traces, not real backend traces.

2. No backend resolution

The backend branch remains unresolved.

3. No semantic interpretation

Candidate relations are treated structurally, not semantically.

4. No empirical claims beyond the offline branch

No claim should be made that OMNIAMIND has already been validated on real model generation traces.


---

Current project state

Analytical branch: operational
Backend branch: unresolved
Offline package state: frozen


---

Meaning of “frozen”

Frozen does not mean perfect. Frozen means:

scope fixed

file set defined

entrypoints defined

outputs defined

boundaries declared

no further expansion required before a new milestone


Any future work should branch from this state rather than silently mutating it.


---

Next valid milestones after freeze

Only these next steps are structurally justified:

1. execute a real backend capture


2. map real traces into the existing proxy pipeline


3. compare synthetic and empirical behavior


4. decide whether v0 or v0.1 survives empirical contact better


5. only then define a later proxy revision



Until then, the current package should be treated as complete in the analytical sense.


---

Minimal conclusion

This manifest freezes OMNIAMIND as an offline analytical package in state:

v0 / v0.1

The package is runnable, internally coherent, and structurally bounded.

It is ready for preservation, inspection, and future empirical connection to real backend traces.