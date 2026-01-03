# OMNIAMIND

**Structural Boundary Stability Diagnostics for Numeric and Symbolic Sequences**  
**Author:** Massimiliano Brighindi (MB-X.01)  
**License:** MIT

---

## Overview

**OMNIAMIND** is a **diagnostic-only structural analysis engine** designed to measure  
**global constraint pressure and boundary stability** in numeric or symbolic sequences.

It does **not**:
- generate text
- evaluate semantic correctness
- perform factorization or divisibility tests
- require training, labels, or embeddings
- depend on language models

OMNIAMIND measures **how strongly a structure is constrained as a whole**.

---

## Core Idea

Any genuinely constrained derivation — mathematical, logical, or algorithmic —  
exhibits **global structural rigidity**.

Plausible but unanchored constructions  
(e.g. hallucination-like outputs or arbitrary composites)  
exhibit **lower global rigidity**, even if locally fluent or well-formed.

OMNIAMIND detects this difference using **multi-base modular interference analysis**.

This is a **structural property**, not a semantic one.

---

## Boundary Interpretation

OMNIAMIND does **not** treat boundaries as reactive or observer-dependent entities.

The probing process does not alter, negotiate, or adapt the boundary.  
It merely exposes how the boundary constrains admissible structure.

In this framework:

- the boundary is **revealed**, not constructed  
- stability is **measured**, not induced  
- probing does **not** generate feedback, meaning, or interpretation  

Any observed rigidity is interpreted as an **intrinsic property of the structure**,  
not as an artifact of observation.

---

## What OMNIAMIND Measures

Given a deterministic sequence of integers  
(or tokens mapped deterministically to integers), OMNIAMIND:

- evaluates interference across many modular bases  
- aggregates normalized phase coherence  
- outputs a **rigidity score** ∈ [0, 1]

Interpretation:

- **High score** → strong global constraints / stable boundary  
- **Low score** → weak or absent constraints

This measures **structure**, not correctness or truth.

---

## Hallucination Probe (Diagnostic)

The included hallucination probe compares:

- **A:** anchored / constrained derivations  
- **B:** plausible but unanchored claims  

Example result (representative run):

mean_norm A ≈ 0.63 mean_norm B ≈ 0.34

median_norm A ≈ 0.73 median_norm B ≈ 0.34

This separation emerges **without semantics, embeddings, prompts, or training**.

---

## Included Benchmarks

- `omniamind_rsa_weighted.py`  
  Structural separation between RSA-like semiprimes and random composites  
  of comparable magnitude.

- `omniamind_primes_scan.py`  
  Constraint signatures in prime distributions via modular interference.

- `omniamind_hallucination_probe.py`  
  Rigidity contrast between constrained derivations and hallucination-like text.

All benchmarks are:

- deterministic  
- reproducible  
- model-agnostic  
- diagnostic-only  

---

## Relationship to OMNIA

OMNIAMIND complements **OMNIA**:

- **OMNIA** → measures semantic, causal, and temporal instability under transformation  
- **OMNIAMIND** → measures internal structural constraint pressure and boundary stability  

Together they form a **dual diagnostic layer** for hallucination and instability detection.

---

## Intended Use

- Pre-decision hallucination detection  
- Structural validation layer for LLM outputs  
- Research on constraint-based diagnostics  
- Model-agnostic safety instrumentation  

OMNIAMIND does **not** replace reasoning.  
It measures whether **global constraint actually occurred**.

---

## Status

Experimental but stable.  
Designed for research and diagnostic integration.

No training.  
No prompts.  
No heuristics.

Only structure.

---

## Citation

If you reference this work:

> Brighindi, M. — *OMNIAMIND: Structural Boundary Stability Diagnostics* (2025)

---

## Disclaimer

OMNIAMIND is **not a truth oracle**.  
It detects **absence of structural constraint**, not factual falsity.

Use as a semantic classifier or fact checker is discouraged.


.