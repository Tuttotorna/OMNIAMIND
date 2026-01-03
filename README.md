# OMNIAMIND

**Structural Rigidity Diagnostics for Numeric and Symbolic Sequences**  
**Author:** Massimiliano Brighindi (MB-X.01)  
**License:** MIT

---

## Overview

**OMNIAMIND** is a **diagnostic-only structural analysis engine** designed to measure
**global constraint pressure and rigidity** in numeric or symbolic sequences.

It does **not**:
- generate text
- evaluate semantic correctness
- require training or labels
- depend on language models or embeddings

OMNIAMIND measures **how strongly a sequence is internally constrained**.

---

## Core Idea

Any genuinely constrained derivation (mathematical, logical, algorithmic)
exhibits **global structural rigidity**.

Plausible but unanchored constructions (hallucination-like outputs)
exhibit **lower global rigidity**, even if locally fluent.

OMNIAMIND detects this difference using **modular interference analysis**.

---

## What OMNIAMIND Measures

Given a sequence of integers (or tokens mapped deterministically to integers):

- evaluates interference across many modular bases
- aggregates normalized phase coherence
- outputs a **rigidity score** ∈ [0, 1]

High score → strong global constraints  
Low score → weak or missing constraints

This is **structure**, not meaning.

---

## Hallucination Probe (Diagnostic)

The included hallucination probe compares:

- **A:** anchored / constrained derivations  
- **B:** plausible but unanchored claims

### Example Result

mean_norm A ≈ 0.63 mean_norm B ≈ 0.34

median_norm A ≈ 0.73 median_norm B ≈ 0.34

This separation emerges **without semantics, embeddings, or training**.

---

## Included Benchmarks

- `omniamind_rsa_weighted.py`  
  Structural separation between RSA-like and composite numbers.

- `omniamind_primes_scan.py`  
  Constraint signatures in prime distributions.

- `omniamind_hallucination_probe.py`  
  Rigidity difference between constrained derivations and hallucination-like text.

All benchmarks are:
- deterministic
- reproducible
- model-agnostic

---

## Relationship to OMNIA

OMNIAMIND complements **OMNIA**:

- **OMNIA** → measures semantic / causal instability across transformations
- **OMNIAMIND** → measures internal structural constraint pressure

Together they form a **dual hallucination diagnostic layer**.

---

## Intended Use

- Pre-decision hallucination detection
- Structural validation layer for LLM outputs
- Research on constraint-based truth diagnostics
- Model-agnostic safety instrumentation

OMNIAMIND does **not** replace reasoning.
It measures whether reasoning **actually occurred**.

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

> Brighindi, M. — *OMNIAMIND: Structural Rigidity Diagnostics* (2025)

---

## Disclaimer

OMNIAMIND is **not a truth oracle**.  
It detects **absence of constraint**, not factual falsity.

Misuse as a semantic classifier is discouraged.

