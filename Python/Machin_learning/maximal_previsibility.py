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
    """Numerically solve the Fano inequality to find Pmax. Returns NaN if no valid solution."""
    if N <= 1 or S <= 0:
        return 1.0 if S == 0 else np.nan
    S_max = np.log2(N)
    if S >= S_max:
        return 1.0 / N
    try:
        p_max = brentq(fano_equation, 1e-10, 1 - 1e-10, args=(S, N))
        return p_max
    except ValueError:
        return np.nan

# ── Load data from numpy file ─────────────────────────────────────────────────
raw = np.load(INPUT_DIR / 'user_entropies_no_merge.npy', allow_pickle=True).item()

rows = []
for id_user, values in raw.items():
    day, user_entropy, user_relative_entropy, num_record = values
    rows.append({'id_user': id_user, 'day': day,
                 'entropy': user_entropy, 'n_stations': num_record})

df = pd.DataFrame(rows)

# ── Compute Pmax ──────────────────────────────────────────────────────────────
df['pmax'] = df.apply(
    lambda row: compute_pmax(row['entropy'], row['n_stations']),
    axis=1
)

# ── Statistics ────────────────────────────────────────────────────────────────
print(df['pmax'].describe())
print(f"\nMean Pmax   : {df['pmax'].mean():.3f}")
print(f"Median Pmax : {df['pmax'].median():.3f}")
print(f"% users > 0.8 : {(df['pmax'] > 0.8).mean() * 100:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
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
