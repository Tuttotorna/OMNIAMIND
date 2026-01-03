"""
OMNIAMIND — RSA Structural Ridge (Weighted Interference)

Diagnostic-only structural metric.
No factorization. No divisibility-based features (except for generating labels).
Compares RSA-like semiprimes (p*q) vs random composites of similar scale.

Goal: detect a structural ridge of semiprimality via global modular interference.
Author: Massimiliano Brighindi (MB-X.01 / OMNIAMIND)
License: MIT
"""

from __future__ import annotations

import random
import numpy as np


def interference_signature_weighted(n: int, max_k: int = 1024) -> float:
    """
    Weighted modular interference signature.

    I_w(n) = sum_{k=2..K} |sin(2π n / k)| / log(k+1)

    Intuition:
    - non-averaging, global "interference" across moduli
    - log-weight reduces trivial small-k dominance
    - exposes a semiprime ridge (statistical separation) without factorization
    """
    ks = np.arange(2, max_k + 1, dtype=np.float64)
    w = 1.0 / np.log(ks + 1.0)
    return float(np.sum(w * np.abs(np.sin(2 * np.pi * n / ks))))


def is_prime(x: int) -> bool:
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    r = int(x ** 0.5)
    for d in range(3, r + 1, 2):
        if x % d == 0:
            return False
    return True


def primes_upto(n: int = 2000) -> list[int]:
    return [i for i in range(2, n + 1) if is_prime(i)]


def random_composite(low: int, high: int) -> int:
    while True:
        x = random.randint(low, high)
        if x > 3 and not is_prime(x):
            return x


def normalize_01(values: list[float]) -> list[float]:
    vals = np.array(values, dtype=np.float64)
    vmin, vmax = float(vals.min()), float(vals.max())
    denom = (vmax - vmin) + 1e-12
    return [float((v - vmin) / denom) for v in values]


def main() -> None:
    # Frozen params for reproducibility
    random.seed(0)

    MAX_K = 1024
    SAMPLES = 40
    PRIME_POOL_MAX = 2000
    PRIME_OFFSET = 50  # avoid tiny primes

    PR = primes_upto(PRIME_POOL_MAX)
    if len(PR) <= PRIME_OFFSET + 10:
        raise RuntimeError("Prime pool too small; increase PRIME_POOL_MAX.")

    # Build RSA-like semiprimes
    rsa_items: list[tuple[str, int]] = []
    for _ in range(SAMPLES):
        p = random.choice(PR[PRIME_OFFSET:])
        q = random.choice(PR[PRIME_OFFSET:])
        rsa_items.append(("RSA", p * q))

    rsa_ns = [n for _, n in rsa_items]
    low, high = min(rsa_ns), max(rsa_ns)

    # Matched random composites in same magnitude range
    comp_items: list[tuple[str, int]] = []
    for _ in range(SAMPLES):
        comp_items.append(("COMP", random_composite(low, high)))

    items = rsa_items + comp_items

    # Score
    raw_scores = []
    for tag, n in items:
        sc = interference_signature_weighted(n, MAX_K)
        raw_scores.append(sc)

    norm_scores = normalize_01(raw_scores)

    rows = [(items[i][0], items[i][1], norm_scores[i]) for i in range(len(items))]

    rsa_scores = [sc for tag, _, sc in rows if tag == "RSA"]
    comp_scores = [sc for tag, _, sc in rows if tag == "COMP"]

    print("OMNIAMIND RSA ridge test — weighted")
    print("samples per class:", SAMPLES)
    print("mean_norm RSA :", round(float(np.mean(rsa_scores)), 6))
    print("mean_norm COMP:", round(float(np.mean(comp_scores)), 6))
    print("median_norm RSA :", round(float(np.median(rsa_scores)), 6))
    print("median_norm COMP:", round(float(np.median(comp_scores)), 6))

    rows_sorted = sorted(rows, key=lambda x: x[2])

    print("\nLowest 10 (tag, n, score_norm):")
    for r in rows_sorted[:10]:
        print(r)

    print("\nHighest 10 (tag, n, score_norm):")
    for r in rows_sorted[-10:]:
        print(r)


if __name__ == "__main__":
    main()