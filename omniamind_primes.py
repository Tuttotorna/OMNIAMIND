import numpy as np

def structural_signature(n, max_k=64):
    """
    OMNIAMIND signature:
    measures structural resistance under simultaneous decompositions
    """
    sig = 0.0
    for k in range(2, max_k + 1):
        sig += np.gcd(n, k) / k
    return sig

def omniamind_primes(limit=200):
    """
    Primes are numbers with minimal structural collapsibility
    across all k simultaneously
    """
    scores = {}
    for n in range(2, limit + 1):
        scores[n] = structural_signature(n)

    # Normalize
    values = np.array(list(scores.values()))
    values = (values - values.min()) / (values.max() - values.min())

    return scores, values

if __name__ == "__main__":
    scores, norm = omniamind_primes(200)
    for n, v in scores.items():
        if v < 0.25:  # structural minima
            print(n, round(v, 4))