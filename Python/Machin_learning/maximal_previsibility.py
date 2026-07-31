import numpy as np
from scipy.optimize import brentq
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MAIN_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = MAIN_DIR / "results/numpy"

def fano_equation(p, S, N):
    """Fano equation: find p such that S = H(p) + (1-p)*log2(N-1)"""
    if p <= 0 or p >= 1:
        return float('inf')
    # Binary entropy H(p)
    H_p = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    if N <= 1:
        return float('inf')
    return H_p + (1 - p) * np.log2(N - 1) - S

def compute_pmax(S, N):
    """Numerically solve the Fano inequality to find Pmax. Returns NaN if no valid solution.

    The root is bracketed on [1/N, 1], not on [0, 1]: the Fano function
    H(p) + (1-p)*log2(N-1) peaks at p = 1/N (where it equals log2(N)) and then
    decreases to 0 at p = 1. Since 0 <= S <= log2(N), that bracket always has a
    sign change, and it selects the meaningful root (predictability at least as
    good as random guessing). Bracketing on [0, 1] instead makes brentq fail
    whenever S > log2(N-1) - i.e. always for N = 2 - and silently return NaN.
    """
    if N <= 1 or S <= 0:
        return 1.0 if S == 0 else np.nan
    S_max = np.log2(N)
    if S >= S_max:
        return 1.0 / N
    try:
        p_max = brentq(fano_equation, 1.0 / N, 1 - 1e-10, args=(S, N))
        return p_max
    except ValueError:
        return np.nan

def load_user_entropies(exclude_zero_entropy=True):
    """Load the per-user entropies. Returns a DataFrame with columns:
    id_user, day, entropy, n_records, n_states.

    Users with zero entropy are dropped by default: they never left a single
    cell, so they are trivially predictable and Fano hands them Pmax = 1 by
    convention. Same convention as plot_hist_entropy.py, which already filters
    on S_unc != 0.

    Split out from load_pmax_dataframe() so that callers who only need the
    user filter (e.g. comparing prediction methods) do not pay for ~1.4M
    Fano solves they will not use."""
    raw = np.load(INPUT_DIR / 'user_entropies_no_merge.npy', allow_pickle=True).item()

    rows = []
    for id_user, values in raw.items():
        if len(values) < 5:
            raise ValueError(
                "user_entropies_no_merge.npy predates the Pmax fix and has no "
                "distinct-state count - re-run Machin_learning/transition_emtropy.py "
                "to regenerate it."
            )
        day, user_entropy, user_relative_entropy, num_record, n_states = values
        rows.append({'id_user': id_user, 'day': day, 'entropy': user_entropy,
                     'n_records': num_record, 'n_states': n_states})

    df = pd.DataFrame(rows)
    if exclude_zero_entropy:
        df = df[df['entropy'] > 0].reset_index(drop=True)
    return df


def load_pmax_dataframe(exclude_zero_entropy=True):
    """load_user_entropies() plus a 'pmax' column.

    N in the Fano inequality is n_states - the number of distinct states the
    entropy was computed over (distinct cells, plus 'outside' when the user left
    the area) - and NOT n_records, which counts repeated visits and would
    inflate Pmax."""
    df = load_user_entropies(exclude_zero_entropy=exclude_zero_entropy)
    df['pmax'] = df.apply(
        lambda row: compute_pmax(row['entropy'], row['n_states']),
        axis=1
    )
    return df


if __name__ == "__main__":
    # ── Load data from numpy file ─────────────────────────────────────────────
    df = load_pmax_dataframe()

    # ── Statistics ─────────────────────────────────────────────────────────────
    print(df['pmax'].describe())
    print(f"\nMean Pmax   : {df['pmax'].mean():.3f}")
    print(f"Median Pmax : {df['pmax'].median():.3f}")
    print(f"% users > 0.8 : {(df['pmax'] > 0.8).mean() * 100:.1f}%")

    # ── Plot ───────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.hist(df['pmax'].dropna(), bins=100, color='steelblue', edgecolor='white')
    plt.axvline(df['pmax'].mean(), color='red', linestyle='--',
                label=f"Mean: {df['pmax'].mean():.2f}")
    plt.axvline(0.93, color='orange', linestyle='--', label="Song et al.: 0.93")
    plt.xlabel("Maximum Predictability $P^{max}$")
    plt.ylabel("Number of Users")
    plt.title("Distribution of the Predictability Upper Bound (Fano)")
    plt.legend()
    plt.tight_layout()
    plt.show()
