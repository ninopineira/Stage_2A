# Replication of Song et al. (2010), figure 2, on Database/no_duplicate — for the whole
# day and for three periods of the day, with ONE set of definitions:
#
#   N      : number of distinct states (cells + 'outside' if the user left the area),
#            as returned by transition_emtropy.entropy_for_user — NOT the number of
#            records (that was the old bug, which inflated Pmax: 0.78 instead of 0.58)
#   S_unc  : uncorrelated entropy, transition_emtropy.entropy_for_user
#   S_rand : log2(N)
#   S_rel  : S_unc / S_rand  (the definition used in the report; the older scripts
#            stored S_unc / N instead)
#   Pmax   : root of Fano's equation S = H(P) + (1 - P) log2(N - 1) on [1/N, 1]
#
# Population: user-days with non-zero entropy only (a zero-entropy user never left a
# single state; Fano hands them Pmax = 1 by convention). Same rule for every period.
#
# Periods: each record goes to the period of its own timestamp — Morning [00h, 06h),
# Day [06h, 19h), Evening [19h, 24h]. (transition_entropy_by_period.py split on
# indices and put a user's first record in "Morning" even when it was recorded later.)
#
# Outputs:
#   results/plots/song_replication_no_merge.png        (whole day, 3 panels)
#   results/plots/entropy_by_period_no_merge.png       (3 periods x 3 panels)
#   results/intermediate_result/song_replication_summary.csv
#   results/intermediate_result/song_replication_per_user_day.csv.gz

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

from transition_emtropy import entropy_for_user

MAIN_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = MAIN_DIR / "Database/no_duplicate"
PLOT_DIR = MAIN_DIR / "results/plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = MAIN_DIR / "results/intermediate_result"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERIODS = {                       # name -> [start, end) in seconds since midnight
    "Morning (00h-06h)": (0, 6 * 3600),
    "Day (06h-19h)": (6 * 3600, 19 * 3600),
    "Evening (19h-24h)": (19 * 3600, 24 * 3600 + 1),
}
WHOLE_DAY = "Whole day"


def pmax_fano(S, N, n_iter=60):
    """Vectorised Fano solver: root of H(p) + (1-p) log2(N-1) - S on [1/N, 1].
    The function decreases from log2(N) - S >= 0 to -S <= 0 on that bracket, so a
    bisection always converges (same bracket as maximal_previsibility.compute_pmax)."""
    S = np.asarray(S, dtype=float)
    N = np.asarray(N, dtype=float)
    lo, hi = 1.0 / N, np.ones_like(N)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        h = -(mid * np.log2(mid) + np.where(mid < 1, (1 - mid) * np.log2(np.clip(1 - mid, 1e-300, None)), 0))
        f = h + (1 - mid) * np.log2(np.maximum(N - 1, 1)) - S
        lo = np.where(f > 0, mid, lo)
        hi = np.where(f > 0, hi, mid)
    p = (lo + hi) / 2
    return np.where(S >= np.log2(N), 1.0 / N, p)


def collect():
    rows = []
    files = sorted(DATASET_DIR.glob("*.csv"))
    for file in tqdm.tqdm(files, desc="Days"):
        day = file.name.split("_")[0]
        with open(file, "r", encoding="utf-8", newline="") as f:
            for line in csv.reader(f, delimiter=";"):
                cells = [c for c in line[8::2] if c]
                stamps = [int(t) for t in line[9::2] if t]
                n_rec = len(cells)
                s, _, n = entropy_for_user(cells, stamps)
                rows.append((day, WHOLE_DAY, n_rec, s, n))
                for name, (a, b) in PERIODS.items():
                    idx = [i for i, t in enumerate(stamps) if a <= t < b]
                    if not idx:
                        continue
                    s, _, n = entropy_for_user([cells[i] for i in idx], [stamps[i] for i in idx])
                    rows.append((day, name, len(idx), s, n))
    return pd.DataFrame(rows, columns=["day", "period", "n_records", "S_unc", "N"])


def hist_panels(axes, d, title_suffix=""):
    ax = axes[0]
    ax.hist(d["S_rand"], bins=50, alpha=0.6, color="tomato", density=True, label="$S^{rand}=\\log_2 N$")
    ax.hist(d["S_unc"], bins=50, alpha=0.6, color="steelblue", density=True, label="$S^{unc}$")
    ax.axvline(d["S_rand"].mean(), color="tomato", ls="--", lw=1.5,
               label=f"mean $S^{{rand}}$ = {d['S_rand'].mean():.2f}")
    ax.axvline(d["S_unc"].mean(), color="steelblue", ls="--", lw=1.5,
               label=f"mean $S^{{unc}}$ = {d['S_unc'].mean():.2f}")
    ax.set_xlabel("Entropy (bits)")
    ax.set_ylabel("Density")
    ax.set_title("$P(S^{rand})$ and $P(S^{unc})$" + title_suffix)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.hist(d["S_rel"], bins=50, alpha=0.7, color="mediumseagreen", density=True)
    ax.axvline(d["S_rel"].mean(), color="darkgreen", ls="--", lw=1.5,
               label=f"mean = {d['S_rel'].mean():.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Relative entropy $S^{rel} = S^{unc} / \\log_2 N$")
    ax.set_title("$P(S^{rel})$" + title_suffix)
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.hist(d["pmax"], bins=50, alpha=0.7, color="mediumpurple", density=True)
    ax.axvline(d["pmax"].mean(), color="purple", ls="--", lw=1.5,
               label=f"mean = {d['pmax'].mean():.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Maximum predictability $P^{max}$")
    ax.set_title("$P(P^{max})$" + title_suffix)
    ax.legend(fontsize=7)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    df = collect()
    n_all = (df["period"] == WHOLE_DAY).sum()
    print(f"{n_all:,} user-days; median records {df.loc[df.period == WHOLE_DAY, 'n_records'].median():.0f}, "
          f"mean {df.loc[df.period == WHOLE_DAY, 'n_records'].mean():.1f}")

    df = df[df["S_unc"] > 0].copy()           # non-zero entropy only (hence N >= 2)
    df["S_rand"] = np.log2(df["N"])
    df["S_rel"] = df["S_unc"] / df["S_rand"]
    df["pmax"] = pmax_fano(df["S_unc"].values, df["N"].values)
    df.to_csv(OUT_DIR / "song_replication_per_user_day.csv.gz", sep=";", index=False)

    order = [WHOLE_DAY] + list(PERIODS)
    summary = (df.groupby("period")
                 .agg(n_user_days=("pmax", "size"), median_records=("n_records", "median"),
                      mean_records=("n_records", "mean"), S_rand=("S_rand", "mean"),
                      S_unc=("S_unc", "mean"), S_unc_max=("S_unc", "max"), S_rel=("S_rel", "mean"),
                      pmax=("pmax", "mean"), pmax_median=("pmax", "median"),
                      share_pmax_above_08=("pmax", lambda s: (s > 0.8).mean()))
                 .reindex(order))
    summary.to_csv(OUT_DIR / "song_replication_summary.csv", sep=";")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    hist_panels(axes, df[df.period == WHOLE_DAY])
    fig.suptitle(f"Entropy and predictability distributions — replication of Song et al. (fig. 2)\n"
                 f"{(df.period == WHOLE_DAY).sum():,} user-days with non-zero entropy, N = distinct states",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "song_replication_no_merge.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    for r, name in enumerate(PERIODS):
        d = df[df.period == name]
        hist_panels(axes[r], d, f" — {name}")
        axes[r][0].annotate(f"{len(d):,} user-days", xy=(0.98, 0.5), xycoords="axes fraction",
                            ha="right", fontsize=8, color="gray")
    fig.suptitle("Entropy and predictability by period of the day (non-zero entropy, N = distinct states)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "entropy_by_period_no_merge.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figures saved in", PLOT_DIR)
