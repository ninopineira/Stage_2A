# Part 1d — Lempel-Ziv predictability bound, by MAXIMAL order (Song et al. 2010).
#
# Lempel-Ziv has no fixed order: it estimates the entropy rate from match lengths.
# Here we turn it into a per-"order" curve by capping the MAXIMAL match length at k:
#   Lambda_i^(k) = min(Lambda_i, k+1)  (LZ allowed to use at most k symbols of context).
#   S_LZ^(k) = ( (1/n) sum_i Lambda_i^(k) )^{-1} * log2(n)      [bits/symbol]
#   P^max_LZ^(k) = Fano(S_LZ^(k), n_states).
# So "order k" = the maximal context LZ may exploit. k=1 sees only single-symbol
# repetition; k large approaches full LZ. This mirrors the held-out per-order figure.
#
# CAVEAT: the LZ estimator converges slowly — order (log n)^{-1/2} — so on a single
# short day (median ~44 records) S_LZ is noisy. Read per-day means as reliable, the
# per-user values as indicative. (Kontoyiannis et al. 1998; Song et al., Science 2010.)
#
# Accuracy per order = order-k Markov ACC@1 (conditions on exactly k cells).
#
# Prerequisites: 1_train_test_split.py -> 2b_create_train_ngrams_matrix.py

import csv
import sys
import json
import math
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tqdm

from models import NaiveMarkovChain

MAIN_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = MAIN_DIR / "Database/no_duplicate"

sys.path.insert(0, str(MAIN_DIR / "Python" / "Machin_learning"))
from maximal_previsibility import compute_pmax

TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/ngrams_matrix_train_random.json"

OUTPUT_DIR = MAIN_DIR / "results/predictions/simple_predictor/predictability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = MAIN_DIR / "results/predictions/simple_predictor/plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ORDERS = [1, 2, 3, 4, 5]           # here: maximal LZ order (max match length cap)
MAX_ORDER = max(ORDERS)
N_USERS_PER_DAY = 20000
MIN_RECORDS = 15                   # -> at least 10 prediction points for ACC@1
MAX_RECORDS = 512                  # exclude M2M / automated devices
TEST_FRAC = 0.2
SEED = 67
DAYS_TO_RUN = None                 # None = all 15 days; set an int to test fewer

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_COLORS = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#9467bd',
              4: '#8c564b', 5: '#e377c2', 6: '#bcbd22'}

REC_LABELS = ["few", "med", "many"]
MOB_LABELS = ["low", "med", "high"]
EXAMPLE_COLORS = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
                  '#42d4f4', '#f032e6', '#bfef45', '#000000']


def check_inputs():
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{TRAIN_DATA_PATH} not found.\nRun: 1_train_test_split.py -> 2b_create_train_ngrams_matrix.py")
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"{DATASET_DIR} not found.")


def get_day(filepath):
    return filepath.name.split("_")[0]


def is_weekend(day):
    return datetime.strptime(day, "%Y-%m-%d").weekday() >= 5


# ── Lempel-Ziv entropy rate, capped at a maximal order ───────────────────────

def lz_entropy_maxorder(S, n, max_order):
    """Match-length LZ entropy (bits/symbol) with the match length capped at
    max_order (so Lambda_i^(k) = min(Lambda_i, k+1)). S is the sequence encoded
    as a string, n its length."""
    total = 0
    for i in range(n):
        hist = S[:i]
        L = 1
        while i + L <= n and L <= max_order and S[i:i + L] in hist:
            L += 1
        total += L                       # L = min(Lambda_i, max_order+1)
    avg = total / n
    return math.log2(n) / avg if avg > 0 else float("nan")


def encode_sequence(cells):
    idx = {}
    for c in cells:
        if c not in idx:
            idx[c] = chr(len(idx))
    return "".join(idx[c] for c in cells)


# ── One day ──────────────────────────────────────────────────────────────────

def load_day_test_users(day_file):
    with open(day_file, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    random.Random(SEED).shuffle(rows)
    split = int(len(rows) * (1 - TEST_FRAC))
    return rows[split:][:N_USERS_PER_DAY]


def process_day(day, test_users, markov_models):
    records = []
    for user in test_users:
        n_cells = int(user[7])
        if not (MIN_RECORDS <= n_cells <= MAX_RECORDS):
            continue
        cells = user[8::2]
        n_states = len(set(cells))
        moves = sum(1 for i in range(1, len(cells)) if cells[i] != cells[i - 1])
        move_rate = moves / (len(cells) - 1)

        S = encode_sequence(cells)
        n_pred = n_cells - MAX_ORDER

        rec = {"day": day, "id_user": user[0], "n_records": n_cells,
               "n_states": n_states, "move_rate": move_rate}
        for k in ORDERS:
            s_lz = lz_entropy_maxorder(S, n_cells, k)
            rec[f"pmax_lz_{k}"] = compute_pmax(s_lz, n_states) if not np.isnan(s_lz) else np.nan

            correct = 0
            for start in range(n_pred):
                ctx = "-".join(cells[start + MAX_ORDER - k:start + MAX_ORDER])
                last = cells[start + MAX_ORDER - 1]
                true_next = cells[start + MAX_ORDER]
                top, _ = markov_models[k].predict_next(context=ctx, last=last, top_k=1)
                correct += (top[0][0] == true_next)
            rec[f"acc_{k}"] = correct / n_pred

        records.append(rec)
    return records


# ── Figures ──────────────────────────────────────────────────────────────────

def _calendar_positions(days):
    dates = {d: datetime.strptime(d, "%Y-%m-%d") for d in days}
    fm = min(dates.values())
    fm = fm.fromordinal(fm.toordinal() - fm.weekday())
    pos = {d: ((dates[d].toordinal() - fm.toordinal()) // 7, dates[d].weekday()) for d in days}
    return dates, pos, max(r for r, _ in pos.values()) + 1


def plot_demonstration_calendar(day_summary):
    """Per day, mean P^max_LZ and mean ACC@1 across the maximal orders 1..5."""
    days = sorted(day_summary.keys())
    dates, pos, n_rows = _calendar_positions(days)
    fig, axes = plt.subplots(n_rows, 7, figsize=(19, 3.0 * n_rows),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_visible(False)

    for d in days:
        r, c = pos[d]
        ax = axes[r, c]
        ax.set_visible(True)
        s = day_summary[d]
        ax.plot(ORDERS, [s["pmax_lz"][k] for k in ORDERS], "-s", color="orange",
                lw=2, ms=4, label="$P^{max}_{LZ}$ (max order k)")
        ax.plot(ORDERS, [s["acc"][k] for k in ORDERS], "-o", color="seagreen",
                lw=2, ms=4, label="order-k Markov ACC@1")
        wd = dates[d].weekday()
        ax.set_title(f"{d} ({DAY_NAMES[wd][:3]})  n={s['n']}",
                     color="red" if wd >= 5 else "black", fontsize=9, fontweight="bold")
        ax.set_xticks(ORDERS)
        ax.set_ylim(0, 1)
        ax.grid(True, ls="--", alpha=0.3)

    h, l = axes[pos[days[0]][0], pos[days[0]][1]].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", fontsize=11)
    fig.supxlabel("Maximal LZ order k")
    fig.supylabel("Value (mean over the day's users)")
    fig.suptitle("Lempel-Ziv predictability bound vs accuracy, by maximal order and day",
                 fontsize=14)
    out = PLOT_DIR / "predictability_lz_demonstration_calendar.png"
    fig.savefig(out, dpi=120)
    print(f"Saved -> {out}")
    plt.show()


def _panel_scatter(ax, sub, xcol, ycol, color, jitter=0.006):
    # At a small maximal order, P^max_LZ takes only a few discrete values (Lambda_i
    # is capped at k+1), which shows up as razor-thin vertical lines. A tiny random
    # jitter on the cloud (DISPLAY ONLY; the tracked examples are drawn separately at
    # their true position) spreads those discrete columns into readable bands.
    rng = np.random.default_rng(0)
    x = sub[xcol].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)
    if jitter:
        x = x + rng.uniform(-jitter, jitter, size=len(x))
        y = y + rng.uniform(-jitter, jitter, size=len(y))
    ax.scatter(x, y, s=6, alpha=0.15, color=color, edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, ls="--", alpha=0.3)


def select_examples(sub_day):
    d = sub_day.dropna(subset=["n_records", "move_rate"]).copy()
    if len(d) < 9:
        return [None] * 9
    r1, r2 = d["n_records"].quantile([1 / 3, 2 / 3])
    m1, m2 = d["move_rate"].quantile([1 / 3, 2 / 3])
    def tbin3(v, a, b):
        return 0 if v <= a else (1 if v <= b else 2)
    d["rec_bin"] = d["n_records"].apply(lambda v: tbin3(v, r1, r2))
    d["mob_bin"] = d["move_rate"].apply(lambda v: tbin3(v, m1, m2))
    rec_std = d["n_records"].std() + 1e-9
    mob_std = d["move_rate"].std() + 1e-9
    examples = []
    for rb in range(3):
        for mb in range(3):
            grp = d[(d["rec_bin"] == rb) & (d["mob_bin"] == mb)]
            if grp.empty:
                examples.append(None)
                continue
            tr, tm = grp["n_records"].median(), grp["move_rate"].median()
            dist = (((grp["n_records"] - tr) / rec_std) ** 2
                    + ((grp["move_rate"] - tm) / mob_std) ** 2)
            examples.append(grp.loc[dist.idxmin()])
    return examples


def plot_orders_parallel_lz(df, day, tag):
    """The 5 maximal orders side by side for one day, x = P^max_LZ^(k), y = ACC@1,
    with 9 tracked example users drawn opaque."""
    sub_day = df[df["day"] == day]
    weekday_idx = datetime.strptime(day, "%Y-%m-%d").weekday()
    color = DAY_COLORS[weekday_idx]
    examples = select_examples(sub_day)

    print(f"\nExample users for {tag} ({day}, {DAY_NAMES[weekday_idx]}):")
    print(f"  {'color':<9}{'category':<20}{'id':>12}{'records':>9}{'move_rate':>11}{'n_states':>9}")
    for idx, row in enumerate(examples):
        rb, mb = divmod(idx, 3)
        cat = f"{REC_LABELS[rb]} rec / {MOB_LABELS[mb]} mob"
        if row is None:
            print(f"  {EXAMPLE_COLORS[idx]:<9}{cat:<20}{'(none)':>12}")
            continue
        print(f"  {EXAMPLE_COLORS[idx]:<9}{cat:<20}{row['id_user']:>12}"
              f"{int(row['n_records']):>9}{row['move_rate']:>11.2f}{int(row['n_states']):>9}")

    fig, axes = plt.subplots(1, len(ORDERS), figsize=(4.2 * len(ORDERS), 5.4),
                             sharex=True, sharey=True)
    for ax, k in zip(axes, ORDERS):
        xcol, ycol = f"pmax_lz_{k}", f"acc_{k}"
        sub = sub_day.dropna(subset=[xcol, ycol])
        _panel_scatter(ax, sub, xcol, ycol, color)
        for idx, row in enumerate(examples):
            if row is None or pd.isna(row[xcol]) or pd.isna(row[ycol]):
                continue
            ax.scatter(row[xcol], row[ycol], s=130, color=EXAMPLE_COLORS[idx],
                       edgecolors="black", linewidths=0.8, zorder=5)
        ax.set_title(f"max order {k}", fontsize=11, fontweight="bold")
        ax.set_xlabel("$P^{max}_{LZ}$")
    axes[0].set_ylabel("Markov ACC@1")

    handles = []
    for idx, row in enumerate(examples):
        if row is None:
            continue
        rb, mb = divmod(idx, 3)
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=EXAMPLE_COLORS[idx], markeredgecolor="black",
                              markersize=9, label=f"{REC_LABELS[rb]} rec / {MOB_LABELS[mb]} mob"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)

    wd = DAY_NAMES[weekday_idx]
    fig.suptitle(f"{tag}: {day} ({wd})", fontsize=13)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = PLOT_DIR / f"predictability_lz_parallel_{tag}.png"
    fig.savefig(out, dpi=130)
    print(f"Saved -> {out}")
    plt.show()


if __name__ == "__main__":
    check_inputs()
    t0 = time.time()

    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print("Building order-1..5 Markov models...")
    markov_models = {k: NaiveMarkovChain(order=k, train_data=train_data) for k in ORDERS}

    day_files = sorted(DATASET_DIR.glob("*.csv"))
    if DAYS_TO_RUN is not None:
        day_files = day_files[:DAYS_TO_RUN]

    all_records = []
    day_summary = {}
    for day_file in tqdm.tqdm(day_files, desc="Days"):
        day = get_day(day_file)
        recs = process_day(day, load_day_test_users(day_file), markov_models)
        all_records.extend(recs)
        dfd = pd.DataFrame(recs)
        day_summary[day] = {
            "n": len(dfd),
            "acc": {k: dfd[f"acc_{k}"].mean() for k in ORDERS},
            "pmax_lz": {k: dfd[f"pmax_lz_{k}"].mean() for k in ORDERS},
        }
        acc = " ".join(f"k{k}:{day_summary[day]['acc'][k]:.2f}" for k in ORDERS)
        plz = " ".join(f"k{k}:{day_summary[day]['pmax_lz'][k]:.2f}" for k in ORDERS)
        print(f"  {day} n={len(dfd):5d}  ACC[{acc}]  Pmax_LZ[{plz}]")

    df = pd.DataFrame(all_records)
    per_user_csv = OUTPUT_DIR / "predictability_lz_by_order_per_user.csv"
    df.to_csv(per_user_csv, sep=";", index=False)
    print(f"\nPer-user table -> {per_user_csv}")
    print(f"Compute time: {time.time() - t0:.1f}s")

    plot_demonstration_calendar(day_summary)

    days = sorted(df["day"].unique())
    weekdays = [d for d in days if not is_weekend(d)]
    weekends = [d for d in days if is_weekend(d)]
    if weekdays:
        plot_orders_parallel_lz(df, weekdays[0], "weekday")
    if weekends:
        plot_orders_parallel_lz(df, weekends[0], "weekend")
    else:
        print("No weekend day in the processed range — set DAYS_TO_RUN = None to include weekends.")
