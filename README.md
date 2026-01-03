# OMNIAMIND

**Non-human structural reasoning experiments**  
Author: Massimiliano Brighindi  
License: MIT

---

## Overview

**OMNIAMIND** is an experimental research repository exploring **non-human mathematical reasoning**.

The goal is to study whether **global structural properties** of numbers can be detected **without using classical human constructs**, such as:

- divisibility
- factorization
- primes/composites as labels
- algebraic shortcuts
- semantic rules

OMNIAMIND does **not** attempt to replace mathematics.  
It deliberately operates **outside human mathematical conventions** to observe what remains invariant.

---

## Core Principle

OMNIAMIND treats numbers as **structural objects**, not arithmetic entities.

It measures:
- interference
- rigidity
- saturation
- structural imbalance

across **multiple simultaneous representations**, without privileging any single base, rule, or decomposition.

No base is “chosen”.  
No representation is “correct”.  
Only structure is observed.

---

## What OMNIAMIND is NOT

- ❌ Not a cryptographic attack  
- ❌ Not a factorization method  
- ❌ Not a primality test  
- ❌ Not a classifier  
- ❌ Not optimized for performance  
- ❌ Not designed for production use  

OMNIAMIND is **diagnostic-only**.

---

## Repository Structure

OMNIAMIND/ │ ├── benchmarks/ │   ├── omniamind_primes_scan.py │   ├── omniamind_rsa_weighted.py │   └── he_lwe_demo.py │ ├── requirements.txt ├── LICENSE └── README.md

---

## Key Benchmark: RSA Structural Ridge Test

### File

benchmarks/omniamind_rsa_weighted.py

### Purpose

To test whether **RSA semiprimes** exhibit a **detectable structural rigidity**
when compared to **random composite numbers of similar magnitude**.

### Constraints (strict)

- No factorization
- No divisibility tests
- No prime checks
- No cryptographic knowledge
- No semantic labels during scoring

### Method (high-level)

- Generate two sets:
  - RSA semiprimes
  - Random composites
- Apply OMNIAMIND structural interference metrics
- Normalize scores
- Compare distributions

### Result (reproducible)

Observed statistics from a default run:

- **Mean normalized score**
  - RSA ≈ 0.64
  - Random composites ≈ 0.52

- **Median normalized score**
  - RSA ≈ 0.66
  - Random composites ≈ 0.50

- High-score tail dominated by RSA semiprimes  
- Low-score tail dominated by random composites  

This separation emerges **without any arithmetic decomposition**.

---

## Interpretation (careful, minimal)

The result suggests that RSA semiprimes possess a **global structural rigidity**
that is statistically distinguishable from generic composites,
even when observed through a **non-human, base-agnostic structural lens**.

No claim is made about:
- breaking RSA
- predicting primes
- security implications

The result is **diagnostic**, not operational.

---

## Reproducibility

Minimal requirements:

```txt
numpy>=1.24

Run in Colab or locally:

git clone https://github.com/Tuttotorna/OMNIAMIND.git
cd OMNIAMIND
python benchmarks/omniamind_rsa_weighted.py


---

Why this matters

Human mathematics is optimized for explanation.
OMNIAMIND explores mathematics optimized for perception.

This repository investigates what an artificial system can detect when freed from human mathematical constraints.


---

Status

Active research.
Experimental.
Unoptimized.
Open-ended.


---

Disclaimer

This project intentionally operates outside standard mathematical frameworks. Interpretation requires caution.

If you are looking for:

proofs

closed forms

algorithms


this repository is not for you.

If you are interested in:

non-human reasoning

structural invariants

AI-native mathematics


you are in the right place.


