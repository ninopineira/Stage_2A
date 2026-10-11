# Part 1d (control) — How far is the Lempel-Ziv estimator from the truth on short sequences?
#
# Control experiment quoted with the Lempel-Ziv bound: sequences drawn uniformly at random
# from N_SYMBOLS symbols have a true entropy rate of exactly log2(N_SYMBOLS) bits. We apply
# the very estimator of PREDICTABILITY_lempelziv_bound.py (no order cap) to such sequences of
# increasing length and report how much it falls short.
#
# No prerequisite: nothing is read from Database/ or results/.

import math
import random

import numpy as np

from PREDICTABILITY_lempelziv_bound import lz_entropy_maxorder

N_SYMBOLS = 8                                   # true entropy = log2(8) = 3 bits
LENGTHS = {80: 2000, 800: 200, 8000: 10, 80000: 1}   # sequence length -> number of draws
SEED = 67
ALPHABET = "ABCDEFGH"[:N_SYMBOLS]


if __name__ == "__main__":
    rng = random.Random(SEED)
    truth = math.log2(N_SYMBOLS)
    print(f"True entropy: {truth:.3f} bits ({N_SYMBOLS} equiprobable symbols)\n")
    print(f"{'length':>8} {'draws':>6} {'mean S_LZ':>10} {'std':>7} {'min':>7} {'max':>7} {'shortfall':>10}")
    for n, n_draws in LENGTHS.items():
        values = []
        for _ in range(n_draws):
            S = "".join(rng.choice(ALPHABET) for _ in range(n))
            values.append(lz_entropy_maxorder(S, n, max_order=n))   # max_order=n : no cap
        print(f"{n:>8} {n_draws:>6} {np.mean(values):>10.3f} {np.std(values):>7.3f} "
              f"{min(values):>7.3f} {max(values):>7.3f} {truth - np.mean(values):>10.3f}")
