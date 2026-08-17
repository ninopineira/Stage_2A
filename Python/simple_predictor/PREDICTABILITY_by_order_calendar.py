# Part 1b — Per-order predictability, day by day (scatter calendars).
#
# For each day and each order k = 1..5, per user:
#   - S_cond_k : order-k conditional entropy H(next | k previous cells), estimated
#                on the user's own day  ->  P^max_cond_k = Fano(S_cond_k, n_states)
#   - acc_k    : ACC@1 of a pure order-k Markov (NaiveMarkovChain(order=k)),
#                which conditions on EXACTLY k previous cells -> aligns with S_cond_k.
#
# NOTE on the model choice: we intentionally use NaiveMarkovChain(order=k), not
# VOMM(max_order=k). The VOMM's order convention is off-by-one — its "order 1" is
# the unigram (no context), so VOMM(max_order=1) collapses to predicting the single
# globally-most-frequent cell (ACC@1 ~= 0.015). A pure order-k Markov is the correct
# match to S_cond_k, gives a real order-1 result, and is far faster (dict lookup).
#
# Figures:
#   - one scatter CALENDAR per order (5 files): 15 daily panels, x = P^max_cond_k,
#     y = per-user ACC@1, plus the y = x line.
#   - one figure with the 5 orders side by side for a chosen WEEKDAY, and one for
#     a chosen WEEKEND day.
#
# RAW, UNSMOOTHED (by request): S_cond_k is estimated on a single user-day, so for
# k >= 2 most k-contexts are seen once -> S_cond_k -> 0 -> P^max_cond_k -> 1. The
# scatter clouds will pile up against x = 1 as k grows: that IS the in-sample
# overfitting artefact, shown deliberately.
#
# Prerequisites: 1_train_test_split.py -> 2b_create_train_ngrams_matrix.py

import csv
import sys
import json
import time
import random
from datetime import datetime
from collections import defaultdict, Counter
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

ORDERS = [1, 2, 3, 4, 5]
MAX_ORDER = max(ORDERS)
N_USERS_PER_DAY = 20000
MIN_PREDICTIONS = 10       # per-user accuracy is too noisy below this
TEST_FRAC = 0.2            # per-day 80/20, matching 1_train_test_split.py
SEED = 67
GAP = 4 * 3600 + 600       # a k-context is not allowed to span a gap > 4h10
DAYS_TO_RUN = None         # None = all 15 days (fast now); set an int to test fewer

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# One colour per day of week (shared with the other plot scripts of the project)
DAY_COLORS = {
    0: '#1f77b4',  # Monday
    1: '#ff7f0e',  # Tuesday
    2: '#2ca02c',  # Wednesday
    3: '#9467bd',  # Thursday
    4: '#8c564b',  # Friday
    5: '#e377c2',  # Saturday
    6: '#bcbd22',  # Sunday
}

# 9 highlighted example users on a 3x3 grid: records level x mobility level.
# Mobility = move rate (fraction of steps where the cell changes).
REC_LABELS = ["few", "med", "many"]     # number of records
MOB_LABELS = ["low", "med", "high"]     # mobility (move rate)
EXAMPLE_COLORS = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
                  '#42d4f4', '#f032e6', '#bfef45', '#000000']


def check_inputs():
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{TRAIN_DATA_PATH} not found.\nRun: 1_train_test_split.py -> 2b_create_train_ngrams_matrix.py"
        )
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"{DATASET_DIR} not found.")


def get_day(filepath):
    return filepath.name.split("_")[0]


def is_weekend(day):
    return datetime.strptime(day, "%Y-%m-%d").weekday() >= 5


# ── Order-k conditional entropy on the user's own day ────────────────────────

def conditional_entropy_order_k(cells, stamps, k):
    """S_cond_k = H(next | k previous cells). A (k+1)-window spanning a gap > GAP
    is skipped. Returns NaN if the user has no valid window of that order."""
    n = len(cells)
    if n < k + 1:
        return float("nan")

    ctx_counts = defaultdict(Counter)
    for i in range(k, n):
        if any(stamps[j + 1] - stamps[j] > GAP for j in range(i - k, i)):
            continue
        ctx_counts[tuple(cells[i - k:i])][cells[i]] += 1

    total = sum(sum(c.values()) for c in ctx_counts.values())
    if total == 0:
        return float("nan")

    s = 0.0
    for counter in ctx_counts.values():
        row = sum(counter.values())
        p_ctx = row / total
        h = 0.0
        for cnt in counter.values():
            p = cnt / row
            h -= p * np.log2(p)
        s += p_ctx * h
    return s


# ── One day ──────────────────────────────────────────────────────────────────

def load_day_test_users(day_file):
    """Per-day 80/20 split (seed 67); return the test portion, capped."""
    with open(day_file, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    random.Random(SEED).shuffle(rows)
    split = int(len(rows) * (1 - TEST_FRAC))
    return rows[split:][:N_USERS_PER_DAY]


def process_day(day, test_users, markov_models):
    """Return per-user records: P^max_cond_k and order-k Markov ACC@1 for each k."""
    records = []
    for user in test_users:
        n_cells = int(user[7])
        n_pred = n_cells - MAX_ORDER
        if n_pred < MIN_PREDICTIONS:
            continue
        cells = user[8::2]
        stamps = [int(ts) for ts in user[9::2]]
        n_states = len(set(cells))
        moves = sum(1 for i in range(1, len(cells)) if cells[i] != cells[i - 1])
        move_rate = moves / (len(cells) - 1)

        rec = {"day": day, "id_user": user[0], "n_records": n_cells,
               "n_states": n_states, "move_rate": move_rate}
        for k in ORDERS:
            s_cond = conditional_entropy_order_k(cells, stamps, k)
            rec[f"S_cond_{k}"] = s_cond
            rec[f"pmax_cond_{k}"] = compute_pmax(s_cond, n_states) if not np.isnan(s_cond) else np.nan

            # Same prediction points for every order: a 5-window, true_next after it.
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

def _panel_scatter(ax, sub, k, color="steelblue"):
    ax.scatter(sub[f"pmax_cond_{k}"], sub[f"acc_{k}"], s=6, alpha=0.15,
               color=color, edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, ls="--", alpha=0.3)


def plot_order_calendar(df, k):
    """One calendar (15 daily panels) of the order-k scatter: x=P^max_cond_k, y=ACC@1."""
    days = sorted(df["day"].unique())
    dates = {d: datetime.strptime(d, "%Y-%m-%d") for d in days}
    first_monday = min(dates.values())
    first_monday = first_monday.fromordinal(first_monday.toordinal() - first_monday.weekday())
    position = {d: ((dates[d].toordinal() - first_monday.toordinal()) // 7, dates[d].weekday())
                for d in days}
    n_rows = max(r for r, _ in position.values()) + 1

    fig, axes = plt.subplots(n_rows, 7, figsize=(19, 3.0 * n_rows),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_visible(False)

    for d in days:
        row, col = position[d]
        ax = axes[row, col]
        ax.set_visible(True)
        sub = df[df["day"] == d].dropna(subset=[f"pmax_cond_{k}", f"acc_{k}"])
        wd = dates[d].weekday()
        _panel_scatter(ax, sub, k, DAY_COLORS[wd])
        ax.set_title(f"{d} ({DAY_NAMES[wd][:3]})",
                     color="red" if wd >= 5 else "black", fontsize=9, fontweight="bold")

    fig.supxlabel(f"$P^{{max}}_{{cond}}$ at order {k}")
    fig.supylabel(f"Order-{k} Markov ACC@1")
    fig.suptitle(f"Predictability vs accuracy — order {k}, day by day ", fontsize=14)
    out = PLOT_DIR / f"predictability_scatter_calendar_order{k}.png"
    fig.savefig(out, dpi=110)
    print(f"Saved -> {out}")
    plt.show()


def select_examples(sub_day):
    """Pick 9 example users on a 3x3 grid: (few/med/many records) x (low/med/high
    mobility = move rate). One representative per cell, the user closest to the
    cell's centre. Index = rec_bin*3 + mob_bin. Returns 9 rows (None if a cell is
    empty, e.g. few records + high mobility can be rare)."""
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


def plot_orders_parallel(df, day, tag):
    """The 5 orders side by side for one day, plus 9 tracked example users drawn
    opaque so the same user can be followed from order to order."""
    sub_day = df[df["day"] == day]
    weekday_idx = datetime.strptime(day, "%Y-%m-%d").weekday()
    color = DAY_COLORS[weekday_idx]
    examples = select_examples(sub_day)

    # ── Print the 9 chosen example users ──────────────────────────────────────
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
        sub = sub_day.dropna(subset=[f"pmax_cond_{k}", f"acc_{k}"])
        _panel_scatter(ax, sub, k, color)
        # overlay the 9 tracked users (opaque, black edge)
        for idx, row in enumerate(examples):
            if row is None:
                continue
            xk, yk = row[f"pmax_cond_{k}"], row[f"acc_{k}"]
            if pd.isna(xk) or pd.isna(yk):
                continue
            ax.scatter(xk, yk, s=130, color=EXAMPLE_COLORS[idx],
                       edgecolors="black", linewidths=0.8, zorder=5)
        ax.set_title(f"order {k}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"$P^{{max}}_{{cond}}$")
    axes[0].set_ylabel("Markov ACC@1")

    handles = []
    for idx, row in enumerate(examples):
        if row is None:
            continue
        rb, mb = divmod(idx, 3)
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=EXAMPLE_COLORS[idx], markeredgecolor="black",
                              markersize=9,
                              label=f"{REC_LABELS[rb]} rec / {MOB_LABELS[mb]} mob"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   fontsize=8, frameon=False)

    wd = DAY_NAMES[weekday_idx]
    fig.suptitle(f"{tag}: {day} ({wd}) — 9 tracked users", fontsize=13)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = PLOT_DIR / f"predictability_scatter_parallel_{tag}.png"
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
    for day_file in tqdm.tqdm(day_files, desc="Days"):
        day = get_day(day_file)
        recs = process_day(day, load_day_test_users(day_file), markov_models)
        all_records.extend(recs)
        dfd = pd.DataFrame(recs)
        accs = " ".join(f"k{k}:{dfd[f'acc_{k}'].mean():.2f}" for k in ORDERS)
        pmx = " ".join(f"k{k}:{dfd[f'pmax_cond_{k}'].mean():.2f}" for k in ORDERS)
        print(f"  {day}  n={len(dfd):5d}  ACC[{accs}]  Pmax[{pmx}]")

    df = pd.DataFrame(all_records)
    per_user_csv = OUTPUT_DIR / "predictability_by_order_per_user.csv"
    df.to_csv(per_user_csv, sep=";", index=False)
    print(f"\nPer-user table -> {per_user_csv}")
    print(f"Compute time: {time.time() - t0:.1f}s")

    # ── One scatter calendar per order ───────────────────────────────────────
    for k in ORDERS:
        plot_order_calendar(df, k)

    # ── The 5 orders side by side, for one weekday and one weekend day ────────
    days = sorted(df["day"].unique())
    weekdays = [d for d in days if not is_weekend(d)]
    weekends = [d for d in days if is_weekend(d)]
    if weekdays:
        plot_orders_parallel(df, weekdays[0], "weekday")
    if weekends:
        plot_orders_parallel(df, weekends[0], "weekend")
    else:
        print("No weekend day in the processed range — set DAYS_TO_RUN = None to include weekends.")
