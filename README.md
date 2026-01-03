# OMNIAMIND
Exploratory research on pre-formal structural reasoning and stability diagnostics. This repository contains experimental frameworks (e.g. Hardness Envelope) for analyzing invariance and collapse in complex problem spaces. Not cryptanalysis. Not attacks.
# OMNIAMIND

Exploratory research on **pre-formal structural reasoning** and **stability diagnostics**.

This repository contains experimental frameworks (e.g. **Hardness Envelope**) for analyzing **invariance** and **collapse** in complex problem spaces.

**Not cryptanalysis. Not attacks.**  
This work is diagnostic-only: it measures structural stability under lawful transformations.

## Relationship to LON / OMNIA
- **LON / OMNIA**: stable, model-agnostic structural measurement engine.
- **OMNIAMIND** (this repo): exploratory research layer (quarantined from OMNIA to keep OMNIA strictly “tool-grade”).

## Hardness Envelope (HE) — idea (minimal)
For a hardness assumption problem \(P\), HE measures whether instances remain **structurally stable** under transformations that preserve the problem class.

Output:
- an instability score (Ω_HE)
- a coarse label: ROBUST / BORDERLINE / FRAGILE
- machine-readable reports (JSON)

## Current artifact
### HE-LWE demo (synthetic)
A diagnostic demo for **LWE-like** synthetic instances:
- generates (A, b) with noise
- applies lawful transformations (row/col permutations, conservative centering, sparse sample mixing)
- extracts structural features (SVD conditioning, effective rank, correlations)
- computes **Ω_HE** as dispersion across transforms

This demo is **not** cryptanalysis and **not** a break of any real scheme.

## Quick run
```bash
pip install -r requirements.txt
python benchmarks/he_lwe_demo.py

Output:
prints Ω_HE and a label for a small parameter grid
writes he_lwe_demo_report.json
Roadmap (short)
HE(P) adapters for additional problem families (diagnostic-only)
stability maps over parameter regimes
connect HE outputs back to OMNIA as optional “research artifacts” only
Author: Massimiliano Brighindi
Signature: MB-X.01 / Omniabase±
