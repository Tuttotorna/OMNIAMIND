# benchmarks/he_lwe_demo.py
# OMNIAMIND — Hardness Envelope (HE) demo for synthetic LWE
# Diagnostic-only: measures structural stability under lawful transformations.
# Author: Massimiliano Brighindi (MB-X.01 / Omniabase±)

from __future__ import annotations
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def lift_mod_q(x: np.ndarray, q: int) -> np.ndarray:
    """Deterministic centered lift to [-q/2, q/2]."""
    x = np.asarray(x, dtype=np.int64) % q
    half = q // 2
    x = np.where(x > half, x - q, x)
    return x.astype(np.float64)


def effective_rank(singular_values: np.ndarray, eps: float = 1e-12) -> float:
    """Effective rank via entropy of normalized singular values."""
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / np.sum(s)
    h = -np.sum(p * np.log(p + eps))
    return float(np.exp(h))


def condition_number_from_svd(s: np.ndarray, eps: float = 1e-12) -> float:
    s = np.asarray(s, dtype=np.float64)
    s = s[s > eps]
    if s.size == 0:
        return float("inf")
    return float(s.max() / (s.min() + eps))


def corr_abs_mean(A: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute Pearson correlation between each column of A and b."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    b0 = b - b.mean()
    denom_b = np.linalg.norm(b0) + 1e-12
    cols = A.shape[1]
    acc = 0.0
    for j in range(cols):
        a0 = A[:, j] - A[:, j].mean()
        denom_a = np.linalg.norm(a0) + 1e-12
        c = float(np.dot(a0, b0) / (denom_a * denom_b))
        acc += abs(c)
    return acc / max(cols, 1)


@dataclass(frozen=True)
class LWEParams:
    n: int
    q: int
    m: int
    sigma: float


def sample_discrete_gaussian(m: int, sigma: float) -> np.ndarray:
    # Rounded Gaussian (toy). Diagnostic-only.
    return np.rint(np.random.normal(loc=0.0, scale=sigma, size=m)).astype(np.int64)


def lwe_instance(params: LWEParams) -> Tuple[np.ndarray, np.ndarray]:
    n, q, m, sigma = params.n, params.q, params.m, params.sigma
    A = np.random.randint(0, q, size=(m, n), dtype=np.int64)
    s = np.random.randint(0, q, size=(n,), dtype=np.int64)
    e = sample_discrete_gaussian(m, sigma) % q
    b = (A @ s + e) % q
    return A, b


# -----------------------------
# Lawful transformations (T1–T3)
# -----------------------------

def T_row_permute(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = A.shape[0]
    p = np.random.permutation(m)
    return A[p, :], b[p]


def T_col_permute(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[1]
    p = np.random.permutation(n)
    return A[:, p], b


def T_center_columns(A: np.ndarray, b: np.ndarray, q: int) -> Tuple[np.ndarray, np.ndarray]:
    # Conservative centering in lifted space, then map back mod q.
    A_l = lift_mod_q(A, q)
    col_means = np.rint(A_l.mean(axis=0)).astype(np.int64)
    A2 = (A.astype(np.int64) - col_means) % q
    return A2, b


def T_sample_mix_sparse(A: np.ndarray, b: np.ndarray, q: int, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sparse mixing on sample space via random elementary row ops:
    row_i <- row_i + row_j (mod q).
    Diagnostic-only.
    """
    A2 = A.copy().astype(np.int64)
    b2 = b.copy().astype(np.int64)
    m = A.shape[0]
    for _ in range(m * k):
        i = np.random.randint(0, m)
        j = np.random.randint(0, m - 1)
        if j >= i:
            j += 1
        A2[i, :] = (A2[i, :] + A2[j, :]) % q
        b2[i] = (b2[i] + b2[j]) % q
    return A2, b2


# -----------------------------
# Feature extraction (I1–I4)
# -----------------------------

def features(A: np.ndarray, b: np.ndarray, q: int) -> Dict[str, float]:
    A_l = lift_mod_q(A, q)
    b_l = lift_mod_q(b, q)

    s = np.linalg.svd(A_l, full_matrices=False, compute_uv=False)
    cond = condition_number_from_svd(s)
    erank = effective_rank(s)
    denom = float(min(A_l.shape[0], A_l.shape[1])) if min(A_l.shape) > 0 else 1.0
    erank_ratio = float(erank / denom)

    corr = corr_abs_mean(A_l, b_l)
    b_std = float(np.std(b_l))
    b_entropy_proxy = float(np.log(1.0 + b_std))

    return {
        "cond": float(cond),
        "erank": float(erank),
        "erank_ratio": float(erank_ratio),
        "corr_abs_mean": float(corr),
        "b_std": float(b_std),
        "b_entropy_proxy": float(b_entropy_proxy),
    }


def omega_he(feature_list: List[Dict[str, float]]) -> float:
    """Dispersion-based instability score across transformed instances."""
    if not feature_list:
        return 0.0

    keys = list(feature_list[0].keys())
    X = np.array([[f[k] for k in keys] for f in feature_list], dtype=np.float64)

    # Stabilize conditioning
    for idx, k in enumerate(keys):
        if k == "cond":
            X[:, idx] = np.log1p(X[:, idx])

    eps = 1e-12
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    cv = sd / (np.abs(mu) + eps)
    return float(np.mean(cv))


def run_one(params: LWEParams, trials: int = 6, subsample_fracs: List[float] | None = None) -> Dict:
    if subsample_fracs is None:
        subsample_fracs = [1.0, 0.75]

    A0, b0 = lwe_instance(params)
    q = params.q

    feats_all: List[Dict[str, float]] = []
    tags_all: List[str] = []

    def add(tag: str, A: np.ndarray, b: np.ndarray):
        feats_all.append(features(A, b, q))
        tags_all.append(tag)

    add("base", A0, b0)

    for t in range(trials):
        A, b = T_row_permute(A0, b0); add(f"T1_rowperm_{t}", A, b)
        A, b = T_col_permute(A0, b0); add(f"T1_colperm_{t}", A, b)
        A, b = T_center_columns(A0, b0, q); add(f"T2_center_{t}", A, b)
        A, b = T_sample_mix_sparse(A0, b0, q, k=2); add(f"T3_mix_{t}", A, b)

    # subsampling stability
    m = params.m
    for frac in subsample_fracs:
        if frac >= 0.999:
            continue
        k = max(10, int(m * frac))
        idx = np.random.choice(m, size=k, replace=False)
        add(f"I4_subsample_{frac:.2f}", A0[idx, :], b0[idx])

    score = omega_he(feats_all)
    if score < 0.08:
        label = "ROBUST"
    elif score < 0.18:
        label = "BORDERLINE"
    else:
        label = "FRAGILE"

    return {
        "params": params.__dict__,
        "omega_he": score,
        "label": label,
        "features": list(zip(tags_all, feats_all)),
    }


def main() -> None:
    set_seed(1337)

    grid = [
        LWEParams(n=32, q=12289, m=96,  sigma=2.0),
        LWEParams(n=32, q=12289, m=96,  sigma=1.2),
        LWEParams(n=32, q=12289, m=64,  sigma=1.2),
        LWEParams(n=48, q=12289, m=144, sigma=2.0),
        LWEParams(n=48, q=12289, m=144, sigma=1.2),
    ]

    results = []
    for p in grid:
        r = run_one(p, trials=6)
        results.append(r)
        print(f"{p} -> omega_he={r['omega_he']:.4f}  {r['label']}")

    out = {
        "he_version": "0.1",
        "note": "Diagnostic-only stability envelope for synthetic LWE instances.",
        "results": results,
    }

    with open("he_lwe_demo_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote: he_lwe_demo_report.json")


if __name__ == "__main__":
    main()