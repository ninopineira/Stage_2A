# Part 1e (illustration) — Every entropy of the report computed on ONE small day.
#
# Worked example quoted in the entropy synthesis of the report: the same 22-record day
# (H = home, W = work, S = shop) is given to each entropy used in the project, with the
# very functions of the project, so that their differences can be read on one sequence:
#   S_rand, S_unc                      (visit frequencies, no memory)
#   plug-in S_cond of order 1..3       (PREDICTABILITY_heldout_bound.plugin_entropy_order_k)
#   held-out entropy of order 1..3     (PREDICTABILITY_heldout_bound.heldout_entropy_order_k)
#   Lempel-Ziv, no order cap           (PREDICTABILITY_lempelziv_bound.lz_entropy_maxorder)
# and each one is converted into P^max by the project's Fano solver (compute_pmax).
#
# No prerequisite: nothing is read from Database/ or results/. Prints a table, writes nothing.

import math
from collections import Counter

from PREDICTABILITY_heldout_bound import (compute_pmax, heldout_entropy_order_k,
                                          order_k_points, plugin_entropy_order_k)
from PREDICTABILITY_lempelziv_bound import encode_sequence, lz_entropy_maxorder

TOY_DAY = "H H H H W W W W W S W W W H H S H H H H H H".split()
ORDERS = [1, 2, 3]


def uncorrelated_entropy(cells):
    n = len(cells)
    return -sum(c / n * math.log2(c / n) for c in Counter(cells).values())


if __name__ == "__main__":
    cells = TOY_DAY
    n, n_states = len(cells), len(set(cells))
    stamps = list(range(n))             # one second apart: no gap, no "outside" state
    stay_rate = sum(cells[i] == cells[i + 1] for i in range(n - 1)) / (n - 1)

    print(f"Sequence : {' '.join(cells)}")
    print(f"n = {n} records, N = {n_states} states, counts = {dict(Counter(cells))}, "
          f"immobility rate = {stay_rate:.3f}\n")

    rows = [("S_rand = log2(N)", math.log2(n_states)),
            ("S_unc", uncorrelated_entropy(cells))]
    for k in ORDERS:
        rows.append((f"plug-in S_cond, order {k}", plugin_entropy_order_k(order_k_points(cells, stamps, k))))
    for k in ORDERS:
        rows.append((f"held-out, order {k}", heldout_entropy_order_k(order_k_points(cells, stamps, k), n_states)))
    rows.append(("Lempel-Ziv, no cap", lz_entropy_maxorder(encode_sequence(cells), n, max_order=n)))

    print(f"{'entropy':<28}{'bits':>8}{'P^max':>9}")
    for label, s in rows:
        print(f"{label:<28}{s:>8.3f}{compute_pmax(s, n_states):>9.3f}")

    print()
    for k in ORDERS:
        contexts = Counter(ctx for ctx, _ in order_k_points(cells, stamps, k))
        print(f"order {k}: {sum(contexts.values())} prediction points, {len(contexts)} distinct contexts, "
              f"{sum(1 for v in contexts.values() if v == 1)} seen only once")
