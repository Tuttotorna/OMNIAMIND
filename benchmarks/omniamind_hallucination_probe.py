"""
OMNIAMIND — Hallucination Probe (Diagnostic-only)

Goal:
Detect lower global structural rigidity in hallucinated outputs vs constrained derivations,
WITHOUT semantic models, embeddings, or label-based training.

Method:
1) Map text -> deterministic numeric sequence (token-hash to [1..M])
2) Compute OMNIAMIND interference (weighted) over the sequence
3) Compare distributions for two groups:
   A) constrained/anchored derivations (structured constraints)
   B) plausible-but-unanchored claims (hallucination-like)

This is NOT a truth detector. It is a rigidity / constraint-pressure detector.
Author: Massimiliano Brighindi (MB-X.01 / OMNIAMIND)
License: MIT
"""

from __future__ import annotations

import re
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np


# -----------------------------
# Deterministic text -> integers (base-free)
# -----------------------------

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")

def tokenize(text: str) -> List[str]:
    text = text.strip()
    return TOKEN_RE.findall(text)

def tok_to_int(tok: str, mod: int = 65521) -> int:
    """
    Deterministic token hash mapped to [1..mod-1].
    Uses SHA256, then reduces mod.
    """
    h = hashlib.sha256(tok.encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "big") % mod
    return int(v if v != 0 else 1)

def text_to_seq(text: str, mod: int = 65521, max_len: int = 256) -> np.ndarray:
    toks = tokenize(text)[:max_len]
    seq = np.array([tok_to_int(t, mod=mod) for t in toks], dtype=np.float64)
    return seq


# -----------------------------
# OMNIAMIND interference over sequences
# -----------------------------

def interference_weighted_seq(seq: np.ndarray, max_k: int = 512) -> float:
    """
    Sequence interference:
    Treat token-ints as a signal; measure global modular interference.

    For each k, compute phase sum over elements:
        s_k = sum_i exp(2πj * x_i / k)
    Then measure |s_k| normalized by length (rigidity proxy).
    Weighted by 1/log(k+1) to reduce trivial dominance.
    """
    seq = np.asarray(seq, dtype=np.float64)
    n = int(seq.size)
    if n == 0:
        return 0.0

    ks = np.arange(2, max_k + 1, dtype=np.float64)
    w = 1.0 / np.log(ks + 1.0)

    # Compute rigidity proxy: mean |sum exp(j*phase)| / n across ks
    acc = 0.0
    for k, wk in zip(ks, w):
        phases = (2.0 * np.pi * seq) / k
        s = np.sum(np.cos(phases)) + 1j * np.sum(np.sin(phases))
        acc += wk * (abs(s) / n)

    return float(acc / np.sum(w))


# -----------------------------
# Minimal benchmark set (A vs B)
# -----------------------------

@dataclass(frozen=True)
class Sample:
    label: str  # "A" anchored/constrained, "B" hallucination-like
    text: str

SAMPLES: List[Sample] = [
    # A: anchored derivations / constrained explanations
    Sample("A", "If a = 3 and b = 5, then a+b = 8. If we add 2 more, total is 10."),
    Sample("A", "Given x^2 - 9 = 0, factor as (x-3)(x+3)=0, so x=3 or x=-3."),
    Sample("A", "A triangle with sides 3,4,5 is right-angled because 3^2+4^2=5^2."),
    Sample("A", "To convert 1011 (base 2) to decimal: 1*8 + 0*4 + 1*2 + 1*1 = 11."),
    Sample("A", "If the probability of heads is 0.5, expected heads in 100 flips is 50."),
    Sample("A", "Derivative of x^3 is 3x^2 by the power rule."),
    Sample("A", "If f(x)=2x+1, then f(4)=9. Substitution gives 2*4+1=9."),
    Sample("A", "For n=15, factors are 1,3,5,15. Divisibility by 3: 1+5=6, yes."),
    Sample("A", "Sum 1..n is n(n+1)/2. For n=10 => 10*11/2 = 55."),
    Sample("A", "If y is proportional to x and y=12 when x=3, then y=4x."),

    # B: hallucination-like (plausible style, weak constraints, invented specifics)
    Sample("B", "The derivative of x^3 is 6x because differentiation doubles the exponent."),
    Sample("B", "A 3-4-5 triangle is equilateral since all Pythagorean triples have equal sides."),
    Sample("B", "1011 in binary equals 13 because you add the digits and multiply by two."),
    Sample("B", "The sum from 1 to 10 is 60 due to the average being 6."),
    Sample("B", "If x^2-9=0 then x must be 9 because negatives are not valid roots here."),
    Sample("B", "Probability of heads in 100 flips is always exactly 52 on average for fair coins."),
    Sample("B", "If f(x)=2x+1 then f(4)=7 because you subtract 1 before multiplying."),
    Sample("B", "15 is prime because it cannot be divided by any even number."),
    Sample("B", "If y is proportional to x and y=12 at x=3 then y=3x, it follows by symmetry."),
    Sample("B", "Adding 3 and 5 gives 9 because addition carries one by default in base ten."),
]


# -----------------------------
# Run + report
# -----------------------------

def normalize_01(vals: List[float]) -> List[float]:
    a = np.array(vals, dtype=np.float64)
    vmin, vmax = float(a.min()), float(a.max())
    denom = (vmax - vmin) + 1e-12
    return [float((v - vmin) / denom) for v in vals]

def main() -> None:
    MOD = 65521
    MAX_LEN = 256
    MAX_K = 512

    rows: List[Dict] = []
    raw = []

    for s in SAMPLES:
        seq = text_to_seq(s.text, mod=MOD, max_len=MAX_LEN)
        score = interference_weighted_seq(seq, max_k=MAX_K)
        raw.append(score)
        rows.append({
            "label": s.label,
            "score_raw": float(score),
            "len_tokens": int(seq.size),
            "text": s.text,
        })

    norm = normalize_01(raw)
    for i in range(len(rows)):
        rows[i]["score_norm"] = float(norm[i])

    A = [r["score_norm"] for r in rows if r["label"] == "A"]
    B = [r["score_norm"] for r in rows if r["label"] == "B"]

    print("OMNIAMIND Hallucination Probe (diagnostic-only)")
    print("samples A (anchored):", len(A), "samples B (hallucination-like):", len(B))
    print("mean_norm A:", round(float(np.mean(A)), 6))
    print("mean_norm B:", round(float(np.mean(B)), 6))
    print("median_norm A:", round(float(np.median(A)), 6))
    print("median_norm B:", round(float(np.median(B)), 6))

    # Show extremes
    rows_sorted = sorted(rows, key=lambda r: r["score_norm"])
    print("\nLowest 6 rigidity (label, score_norm, len_tokens):")
    for r in rows_sorted[:6]:
        print(r["label"], round(r["score_norm"], 6), r["len_tokens"])

    print("\nHighest 6 rigidity (label, score_norm, len_tokens):")
    for r in rows_sorted[-6:]:
        print(r["label"], round(r["score_norm"], 6), r["len_tokens"])

    out = {
        "artifact": "OMNIAMIND_HALLUCINATION_PROBE_v0.1",
        "params": {"mod": MOD, "max_len": MAX_LEN, "max_k": MAX_K},
        "summary": {
            "mean_norm_A": float(np.mean(A)),
            "mean_norm_B": float(np.mean(B)),
            "median_norm_A": float(np.median(A)),
            "median_norm_B": float(np.median(B)),
        },
        "rows": rows,
    }

    with open("omniamind_hallucination_probe.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nWrote: omniamind_hallucination_probe.json")


if __name__ == "__main__":
    main()