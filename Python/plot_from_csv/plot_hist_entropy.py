import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pathlib import Path

MAIN_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = MAIN_DIR / "results/numpy"
OUTPUT_DIR = MAIN_DIR / "results/plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
COL_USER    = "user_id"
COL_STATION = "station"
COL_EXCLUDE = "outside"  # station to exclude (None to keep it)

# ─────────────────────────────────────────────────────────────────────────────
# 1. METRICS PER USER
# ─────────────────────────────────────────────────────────────────────────────
def compute_entropy_metrics(df, col_user, col_station, exclude_station=None):
    """
    For each user computes:
      - N      : number of distinct stations visited
      - S_rand : log2(N)
      - S_unc  : uncorrelated Shannon entropy
      - S_rel  : S_unc / log2(N)  (relative entropy)
      - pmax   : Fano predictability upper bound
    """
    if exclude_station:
        df = df[df[col_station] != exclude_station]

    results = []
    for user, group in df.groupby(col_user):
        counts = group[col_station].value_counts()
        N      = len(counts)
        if N < 2:
            continue

        probs  = counts / counts.sum()
        S_unc  = -np.sum(probs * np.log2(probs))
        S_rand = np.log2(N)
        S_rel  = S_unc / S_rand

        results.append({
            "user":   user,
            "N":      N,
            "S_rand": S_rand,
            "S_unc":  S_unc,
            "S_rel":  S_rel,
            "pmax":   compute_pmax(S_unc, N),
        })

    return pd.DataFrame(results)


def compute_pmax(S, N):
    """Numerically solve the Fano inequality to find Pmax."""
    if N <= 1 or S <= 0:
        return 1.0
    if S >= np.log2(N):
        return 1.0 / N

    def fano(p):
        if p <= 0 or p >= 1:
            return float('inf')
        H_p = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        return H_p + (1 - p) * np.log2(N - 1) - S

    try:
        return brentq(fano, 1e-10, 1 - 1e-10)
    except ValueError:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 2. HISTOGRAMS
# ─────────────────────────────────────────────────────────────────────────────
def plot_histograms(metrics_df):
    """Reproduces figures 2A and 2B from Song et al. on our data."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Entropy and Predictability Distributions",
                 fontsize=13, fontweight="bold")

    # ── Fig. 2A left: P(S_rand), P(S_unc) ────────────────────────────────────
    ax = axes[0]
    ax.hist(metrics_df["S_rand"], bins=50, alpha=0.6,
            color="tomato",    label="$S^{rand} = \\log_2(N)$", density=True)
    ax.hist(metrics_df["S_unc"],  bins=50, alpha=0.6,
            color="steelblue", label="$S^{unc}$ (Shannon)",      density=True)
    ax.axvline(metrics_df["S_rand"].mean(), color="tomato",
               linestyle="--", linewidth=1.5,
               label=f"Mean $S^{{rand}}$ = {metrics_df['S_rand'].mean():.2f}")
    ax.axvline(metrics_df["S_unc"].mean(), color="steelblue",
               linestyle="--", linewidth=1.5,
               label=f"Mean $S^{{unc}}$ = {metrics_df['S_unc'].mean():.2f}")
    ax.set_xlabel("Entropy (bits)")
    ax.set_ylabel("Density")
    ax.set_title("$P(S^{rand})$ and $P(S^{unc})$\n")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Fig. 2A right: P(S_rel) ───────────────────────────────────────────────
    ax = axes[1]
    ax.hist(metrics_df["S_rel"], bins=50, alpha=0.7,
            color="mediumseagreen", density=True)
    ax.axvline(metrics_df["S_rel"].mean(), color="darkgreen",
               linestyle="--", linewidth=1.5,
               label=f"Mean = {metrics_df['S_rel'].mean():.2f}")
    ax.set_xlabel("Relative entropy $S^{unc} / \\log_2(N)$")
    ax.set_ylabel("Density")
    ax.set_title("$P(S^{rel})$\nRelative Entropy")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Fig. 2B: P(Pmax) ──────────────────────────────────────────────────────
    ax = axes[2]
    pmax_clean = metrics_df["pmax"].dropna()
    ax.hist(pmax_clean, bins=50, alpha=0.7,
            color="mediumpurple", density=True)
    ax.axvline(pmax_clean.mean(), color="purple",
               linestyle="--", linewidth=1.5,
               label=f"Mean = {pmax_clean.mean():.2f}")
    ax.set_xlabel("Maximum predictability $P^{max}$")
    ax.set_ylabel("Density")
    ax.set_title("$P(P^{max})$\n")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "entropy_histograms.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved: entropy_histograms.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(metrics_df):
    print("\n── Metrics summary ───────────────────────────────────────")
    for col, label in [("S_rand", "S_rand"), ("S_unc", "S_unc"),
                       ("S_rel",  "S_rel"),  ("pmax",  "Pmax")]:
        s = metrics_df[col].dropna()
        print(f"  {label:8s} | mean={s.mean():.3f}  med={s.median():.3f}"
              f"  std={s.std():.3f}  min={s.min():.3f}  max={s.max():.3f}")
    print(f"\n  % users with Pmax > 0.8  : {(metrics_df['pmax'] > 0.8).mean() * 100:.1f}%")
    print(f"  % users with Pmax > 0.93 : {(metrics_df['pmax'] > 0.93).mean() * 100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Load data from numpy file ─────────────────────────────────────────────
    raw = np.load(INPUT_DIR / 'user_entropies_no_merge.npy', allow_pickle=True).item()

    rows = []
    for id_user, values in raw.items():
        day, user_entropy, user_relative_entropy, num_record = values
        N = num_record
        S_unc = user_entropy
        S_rand = np.log2(N) if N > 1 else np.nan
        if S_unc != 0 :
            rows.append({
                "user":   id_user,
                "N":      N,
                "S_rand": S_rand,
                "S_unc":  S_unc,
                "S_rel":  user_relative_entropy,
                "pmax":   compute_pmax(S_unc, N),
            })

    metrics = pd.DataFrame(rows)
    print(f"✓ {len(metrics)} users processed")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(metrics)

    # ── Histograms ────────────────────────────────────────────────────────────
    plot_histograms(metrics)

    # ── Export ────────────────────────────────────────────────────────────────
    #metrics.to_csv(OUTPUT_DIR / "user_metrics.csv", sep=";", index=False)
    #print("Metrics exported: user_metrics.csv")
