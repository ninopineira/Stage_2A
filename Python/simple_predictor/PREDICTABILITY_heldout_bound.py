# Part 1c — Held-out (cross-validated) entropy as the predictability bound.
#
# Fixes the in-sample collapse of the plug-in order-k entropy (which makes
# P^max_cond -> 1 for k >= 2 because each k-context is seen once on a single day).
#
# Method (per user, per order k):
#   - Build the order-k prediction points (context of k cells -> next cell),
#     skipping windows that span a gap > 4h10.
#   - K-fold cross-validation over these points. For each fold, estimate the
#     smoothed conditional distribution p_hat(next | context) on the OTHER folds
#     (Laplace add-alpha over the user's alphabet; back off to the unigram when
#     the k-context is unseen in the training folds), then measure the surprise on
#     the held-out fold:
#         S_heldout_k = - (1/M) sum_i log2 p_hat(x_i | context_i)          [bits]
#     This is the held-out cross-entropy = an estimate of H(next | k previous),
#     that does NOT collapse to 0 with the order (unseen contexts -> back-off ->
#     bounded surprise; overfitted contexts -> low p_hat -> high surprise).
#   - P^max_heldout_k = Fano(S_heldout_k, n_states).
#
# The plug-in entropy is computed too, only to show side by side that it collapses
# while the held-out estimate stays honest.
#
# Rationale / references: held-out cross-entropy (perplexity) as an upper bound on
# the true entropy is standard in language modelling — Brown et al. 1992; smoothing
# and held-out evaluation — Chen & Goodman 1999. The cross-entropy overestimates
# the true entropy (H(p,p_hat) = H(p) + KL >= H(p)), so P^max_heldout is a
# conservative (safe) ceiling.
#
# Prerequisites: 1_train_test_split.py -> 2b_create_train_ngrams_matrix.py

import csv
import sys
import json
import math
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
MIN_PREDICTIONS = 10
TEST_FRAC = 0.2
SEED = 67
GAP = 4 * 3600 + 600
DAYS_TO_RUN = None          # None = all 15 days; set an int to test fewer

N_FOLDS = 5                 # cross-validation folds for the held-out entropy
ALPHA = 1.0                 # Laplace add-alpha smoothing

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


def order_k_points(cells, stamps, k):
    """(context, next) pairs of order k, skipping windows spanning a gap > GAP."""
    points = []
    for i in range(k, len(cells)):
        if any(stamps[j + 1] - stamps[j] > GAP for j in range(i - k, i)):
            continue
        points.append((tuple(cells[i - k:i]), cells[i]))
    return points


# ── Plug-in entropy (in-sample, collapses) — kept only for the comparison ────

def plugin_entropy_order_k(points):
    if not points:
        return float("nan")
    ctx_counts = defaultdict(Counter)
    for ctx, nxt in points:
        ctx_counts[ctx][nxt] += 1
    total = len(points)
    s = 0.0
    for counter in ctx_counts.values():
        row = sum(counter.values())
        p_ctx = row / total
        h = 0.0
        for cnt in counter.values():
            p = cnt / row
            h -= p * math.log2(p)
        s += p_ctx * h
    return s


# ── Held-out cross-entropy (the fix) ─────────────────────────────────────────

def heldout_entropy_order_k(points, alphabet_size, n_folds=N_FOLDS, alpha=ALPHA):
    """K-fold held-out cross-entropy (bits). Laplace(alpha) over the user's
    alphabet, with back-off to the unigram for unseen contexts. NaN if too few
    points to cross-validate."""
    M = len(points)
    if M < n_folds:
        return float("nan")
    N = max(alphabet_size, 1)

    total_logp = 0.0
    for f in range(n_folds):
        ctx_counts = defaultdict(Counter)
        uni = Counter()
        for j, (ctx, nxt) in enumerate(points):
            if j % n_folds == f:
                continue
            ctx_counts[ctx][nxt] += 1
            uni[nxt] += 1
        uni_total = sum(uni.values())

        for j, (ctx, nxt) in enumerate(points):
            if j % n_folds != f:
                continue
            if ctx in ctx_counts:
                c = ctx_counts[ctx]
                p = (c.get(nxt, 0) + alpha) / (sum(c.values()) + alpha * N)
            elif uni_total > 0:
                p = (uni.get(nxt, 0) + alpha) / (uni_total + alpha * N)
            else:
                p = 1.0 / N
            total_logp += math.log2(p)

    return -total_logp / M


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
            pts = order_k_points(cells, stamps, k)
            s_plugin = plugin_entropy_order_k(pts)
            s_heldout = heldout_entropy_order_k(pts, n_states)
            rec[f"pmax_plugin_{k}"] = compute_pmax(s_plugin, n_states) if not np.isnan(s_plugin) else np.nan
            rec[f"pmax_heldout_{k}"] = compute_pmax(s_heldout, n_states) if not np.isnan(s_heldout) else np.nan

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
    """The headline figure: per day, mean P^max (plug-in vs held-out) and mean
    accuracy across orders 1..5. Shows the plug-in inflating to 1 while the
    held-out bound stays honest and tracks accuracy."""
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
        ax.plot(ORDERS, [s["pmax_plugin"][k] for k in ORDERS], "--s", color="grey",
                lw=1.5, ms=4, label="$P^{max}$ plug-in (in-sample)")
        ax.plot(ORDERS, [s["pmax_heldout"][k] for k in ORDERS], "-s", color="orange",
                lw=2, ms=4, label="$P^{max}$ held-out")
        ax.plot(ORDERS, [s["acc"][k] for k in ORDERS], "-o", color="seagreen",
                lw=2, ms=4, label="Markov ACC@1")
        wd = dates[d].weekday()
        ax.set_title(f"{d} ({DAY_NAMES[wd][:3]})  n={s['n']}",
                     color="red" if wd >= 5 else "black", fontsize=9, fontweight="bold")
        ax.set_xticks(ORDERS)
        ax.set_ylim(0, 1)
        ax.grid(True, ls="--", alpha=0.3)

    h, l = axes[pos[days[0]][0], pos[days[0]][1]].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", fontsize=11)
    fig.supxlabel("Markov order k")
    fig.supylabel("Value (mean over the day's users)")
    fig.suptitle("Held-out bound vs in-sample plug-in vs accuracy, per order and per day",
                 fontsize=14)
    out = PLOT_DIR / "predictability_heldout_demonstration_calendar.png"
    fig.savefig(out, dpi=120)
    print(f"Saved -> {out}")
    plt.show()


def _panel_scatter(ax, sub, xcol, ycol, color):
    ax.scatter(sub[xcol], sub[ycol], s=6, alpha=0.15, color=color, edgecolors="none")
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


def plot_orders_parallel_heldout(df, day, tag):
    """The 5 orders side by side for one day, x = P^max_heldout, y = ACC@1, with
    9 tracked example users drawn opaque."""
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
        xcol, ycol = f"pmax_heldout_{k}", f"acc_{k}"
        sub = sub_day.dropna(subset=[xcol, ycol])
        _panel_scatter(ax, sub, xcol, ycol, color)
        for idx, row in enumerate(examples):
            if row is None:
                continue
            xk, yk = row[xcol], row[ycol]
            if pd.isna(xk) or pd.isna(yk):
                continue
            ax.scatter(xk, yk, s=130, color=EXAMPLE_COLORS[idx],
                       edgecolors="black", linewidths=0.8, zorder=5)
        ax.set_title(f"order {k}", fontsize=11, fontweight="bold")
        ax.set_xlabel("$P^{max}_{heldout}$")
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
    fig.suptitle(f"{tag}: {day} ({wd}) — held-out bound, 9 tracked users", fontsize=13)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    out = PLOT_DIR / f"predictability_heldout_parallel_{tag}.png"
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
            "pmax_plugin": {k: dfd[f"pmax_plugin_{k}"].mean() for k in ORDERS},
            "pmax_heldout": {k: dfd[f"pmax_heldout_{k}"].mean() for k in ORDERS},
        }
        acc = " ".join(f"k{k}:{day_summary[day]['acc'][k]:.2f}" for k in ORDERS)
        pin = " ".join(f"k{k}:{day_summary[day]['pmax_plugin'][k]:.2f}" for k in ORDERS)
        pho = " ".join(f"k{k}:{day_summary[day]['pmax_heldout'][k]:.2f}" for k in ORDERS)
        print(f"  {day} n={len(dfd):5d}  ACC[{acc}]  Pmax_plugin[{pin}]  Pmax_heldout[{pho}]")

    df = pd.DataFrame(all_records)
    per_user_csv = OUTPUT_DIR / "predictability_heldout_per_user.csv"
    df.to_csv(per_user_csv, sep=";", index=False)
    print(f"\nPer-user table -> {per_user_csv}")
    print(f"Compute time: {time.time() - t0:.1f}s")

    plot_demonstration_calendar(day_summary)

    days = sorted(df["day"].unique())
    weekdays = [d for d in days if not is_weekend(d)]
    weekends = [d for d in days if is_weekend(d)]
    if weekdays:
        plot_orders_parallel_heldout(df, weekdays[0], "weekday")
    if weekends:
        plot_orders_parallel_heldout(df, weekends[0], "weekend")
    else:
        print("No weekend day in the processed range — set DAYS_TO_RUN = None to include weekends.")
