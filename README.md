OMNIAMIND — Structural Boundary Stability Diagnostics
Author: Massimiliano Brighindi (MB-X.01)
License: MIT
Status: Experimental but stable
Project lineage: MB-X.01 / OMNIA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18362168.svg)](https://doi.org/10.5281/zenodo.18362168)

## Canonical Ecosystem Map

This repository is part of the **MB-X.01 / OMNIA** ecosystem.

Canonical architecture and full map:
https://github.com/Tuttotorna/lon-mirror/blob/main/ECOSYSTEM.md

---

Overview

OMNIAMIND is a diagnostic-only structural analysis framework designed to measure
boundary stability and global constraint pressure in numeric and symbolic sequences.

It does not:

generate outputs

evaluate semantic correctness

perform factorization or divisibility tests

rely on training, labels, embeddings, or language models

optimize, predict, or decide


OMNIAMIND measures how strongly a structure is constrained as a whole.


---

Core Principle

Any genuinely constrained derivation — mathematical, logical, or algorithmic —
exhibits global structural rigidity.

Plausible but unanchored constructions
(such as arbitrary composites or hallucination-like textual outputs)
exhibit lower global rigidity, even if locally fluent or well-formed.

OMNIAMIND detects this difference using multi-base modular interference probing.

This is a structural property, not a semantic one.


---

Boundary Interpretation

OMNIAMIND treats boundaries as structural objects, not interpretative constructs.

The probing process:

does not alter the boundary

does not induce feedback

does not depend on the observer


It exposes how strongly a boundary constrains admissible structure.

In this framework:

the boundary is revealed, not constructed

stability is measured, not induced

meaning remains external to the system


Any observed rigidity is interpreted as an intrinsic property of the structure,
not as an artifact of observation.


---

What OMNIAMIND Measures

Given a deterministic sequence of integers
(or tokens deterministically mapped to integers), OMNIAMIND:

evaluates interference across many modular bases

aggregates normalized phase coherence

outputs a rigidity score in the interval [0, 1]


Interpretation:

High score → strong global constraints / stable boundary

Low score → weak or absent constraints


OMNIAMIND measures structure, not truth.


---

Benchmarks Included

1. RSA Structural Ridge (Weighted)

File:
benchmarks/omniamind_rsa_weighted.py

Purpose:

Compare RSA-like semiprimes with random composite numbers

Same order of magnitude

No factorization

No divisibility tests


Observation:

RSA semiprimes cluster at higher structural rigidity

Random composites cluster lower

Separation emerges without cryptographic knowledge


This is not an attack on RSA.
It is a diagnostic signal revealing a distinct structural regime.


---

2. Hallucination Probe (Structural)

File:
benchmarks/omniamind_hallucination_probe.py

Purpose:

Compare anchored / constrained derivations
versus plausible but unanchored textual outputs

No semantics

No embeddings

No training


Representative results:

mean_norm (anchored) ≈ 0.63
mean_norm (hallucination-like) ≈ 0.34

median_norm (anchored) ≈ 0.73
median_norm (hallucination-like) ≈ 0.34

The separation emerges purely from structural probing.


---

3. Prime Structure Scan

File:
benchmarks/omniamind_primes_scan.py

Purpose:

Explore constraint signatures in prime distributions

Without primality testing

Without arithmetic decomposition



---

Relationship to OMNIA

OMNIAMIND complements OMNIA:

OMNIA measures semantic, causal, and temporal instability under transformation

OMNIAMIND measures internal structural constraint pressure and boundary stability


Together they form a dual diagnostic layer for instability and hallucination detection.


---

Intended Use

Diagnostic analysis of structured outputs

Pre-decision hallucination detection

Structural validation layer for AI systems

Research on boundary stability and constraint-based diagnostics


OMNIAMIND does not replace reasoning.
It measures whether global constraint actually occurred.


---

Reproducibility

Minimal requirement:

numpy >= 1.24

Run locally:

git clone https://github.com/Tuttotorna/OMNIAMIND.git
cd OMNIAMIND
python benchmarks/omniamind_rsa_weighted.py
python benchmarks/omniamind_hallucination_probe.py

All benchmarks are deterministic and reproducible.


---

Architecture Context (Non-required)

OMNIAMIND does not depend on upstream systems to operate.
However, it is conceptually aligned with the following projects:

LON — Logical Origin Node
Structural and conceptual foundation for invariant-based diagnostics.
https://github.com/Tuttotorna/lon-mirror

OMNIA-LIMIT
Formal boundary artifact certifying when structural measurement can no longer improve discrimination.
https://github.com/Tuttotorna/omnia-limit


These are contextual references, not dependencies.


---

Status

Experimental but stable.
Diagnostic-only.
No roadmap.
No claims beyond observation.


---

Citation

If you reference this work:

> Brighindi, M. — OMNIAMIND: Structural Boundary Stability Diagnostics (2025)




---

Disclaimer

OMNIAMIND is not a truth oracle.
It detects absence of structural constraint, not factual falsity.

Use as a semantic classifier or fact checker is discouraged.