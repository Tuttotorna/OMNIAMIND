# benchmarks/omniamind_primes_scan.py
# OMNIAMIND — Prime structural scan (diagnostic, base-free)
# Author: Massimiliano Brighindi (MB-X.01 / Omniabase±)

from __future__ import annotations
import json
import numpy as np


def structural_signature(n: int, max_k: int = 256) -> float:
    """
    Measures structural collapsibility via simultaneous gcd coupling.
    Lower ~= more "atom-like" under broad decompositions.
    """
    s = 0.0
    for k in range(2, max_k + 1):
        s += np.gcd(n, k) / k
    return float(s)


def scan(limit: int = 5000, max_k: int = 256) -> dict:
    scores = {}
    for n in range(2, limit + 1):
        scores[n] = structural_signature(n, max_k=max_k)

    vals = np.array(list(scores.values()), dtype=np.float64)
    vmin, vmax = float(vals.min()), float(vals.max())
    norm = {n: float((v - vmin) / (vmax - vmin + 1e-12)) for n, v in scores.items()}

    # Rank by "least collapsible" (min norm)
    ranked = sorted(norm.items(), key=lambda x: x[1])

    # Simple baselines for sanity: primes tend to avoid gcd>1 couplings
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

    top = []
    for n, sc in ranked[:200]:
        top.append({
            "n": int(n),
            "score_norm": float(sc),
            "prime": bool(is_prime(int(n))),
        })

    out = {
        "artifact": "OMNIAMIND_PRIMES_SCAN",
        "limit": int(limit),
        "max_k": int(max_k),
        "score_min": vmin,
        "score_max": vmax,
        "top_200_least_collapsible": top,
    }
    return out


def main() -> None:
    out = scan(limit=5000, max_k=256)

    # Print a compact view
    print("Top least-collapsible numbers (n, score_norm, prime):")
    for row in out["top_200_least_collapsible"][:40]:
        print(row["n"], round(row["score_norm"], 6), row["prime"])

    with open("omniamind_primes_scan.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote: omniamind_primes_scan.json")


if __name__ == "__main__":
    main()