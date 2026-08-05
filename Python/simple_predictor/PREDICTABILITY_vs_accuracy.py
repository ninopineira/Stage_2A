# Part 1 — Predictability comparison.
#
# For every test user (a single day, no_duplicate representation, i.e. self-transitions
# A->A are kept — consistent with the train n-grams and the entropy work), in ONE pass,
# we compute both:
#   - the empirical next-cell accuracy of the models (VOMM and order-1 Markov), and
#   - the theoretical predictability ceiling P^max (Fano), from two entropies:
#       * S_unc  : uncorrelated entropy (visit-frequency distribution) -> memoryless ceiling
#       * S_cond : conditional entropy H(next | current)               -> order-1 ceiling
#
# Because everything is computed in the same pass on the same deduplicated sequence,
# accuracy and predictability are aligned per user with no fragile join.
#
# Interpretation (Song et al. framing):
#   P^max_unc <= P^max_cond <= P^max_real   (the last one needs Lempel-Ziv, not done here)
#   - Beating P^max_unc  shows the model exploits order (a memoryless model could not).
#   - Staying under P^max_cond shows how close it gets to the order-1 optimum.
# Note: the comparison uses ACC@1, because P^max bounds the top-1 hit probability only.
#
# Prerequisites (dataset_creation pipeline):
#   1_train_test_split.py  (-> test_random.csv)  ->  2b_create_train_ngrams_matrix.py
#   (3_deduplicate_csv.py is NOT needed here — we use the non-deduplicated test.)

import csv
import sys
import time
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

from models import NaiveMarkovChain, VOMM

MAIN_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = MAIN_DIR / "Database/no_duplicate"

# compute_pmax is the debugged Fano solver from the entropy work; reuse it directly.
sys.path.insert(0, str(MAIN_DIR / "Python" / "Machin_learning"))
from maximal_previsibility import compute_pmax

TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/ngrams_matrix_train_random.json"
# Non-deduplicated test: same representation as the train n-grams AND as the entropy
# work (both on no_duplicate, which keeps self-transitions A->A). Using the
# deduplicated test instead makes "stay" impossible in the ground truth while the
# models' top guess is usually "stay" -> accuracy collapses and S_cond degenerates.
TEST_PATH = MAIN_DIR / "results/predictions/train_test/test_random.csv"
CTX_TOTALS_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/context_totals_train.json"
UNIQUE_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/unique_followers_train.json"
UNIGRAM_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/unigram_train.json"

OUTPUT_DIR = MAIN_DIR / "results/predictions/simple_predictor/predictability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = MAIN_DIR / "results/predictions/simple_predictor/plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTEXT = 5
VOMM_DISCOUNT = 0.9
MIN_PREDICTIONS = 10       # per-user accuracy is too noisy below this
N_USERS = 20000            # eligible users kept (representative once the list is shuffled)
SEED = 67                  # same seed as the project's train/test split
GAP = 4 * 3600 + 600       # "outside" threshold, consistent with transition_emtropy.py

# Predictability bands on P^max_cond, for the stratified view
BANDS = [(0.0, 0.4, "low"), (0.4, 0.7, "medium"), (0.7, 1.01, "high")]


def check_inputs():
    for p in (TRAIN_DATA_PATH, TEST_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.\nRun the dataset_creation pipeline first: "
                "1_train_test_split.py -> 3_deduplicate_csv.py -> 2b_create_train_ngrams_matrix.py"
            )


# ── Entropies (computed on the user's own deduplicated day) ──────────────────

def uncorrelated_entropy(user_cells, user_stamps):
    """S_unc and the number of distinct states. Identical definition to
    Machin_learning/transition_emtropy.py::entropy_for_user, so P^max stays
    consistent with the rest of the entropy work."""
    total_transitions = len(user_cells) - 1
    cells = []
    for c in user_cells:
        if c not in cells:
            cells.append(c)
    cells.append("outside")

    if total_transitions <= 0:
        return 0.0, len(cells) - 1

    nb_gap = sum(1 for i in range(total_transitions)
                 if user_stamps[i + 1] - user_stamps[i] > GAP)
    n_states = (len(cells) - 1) + (1 if nb_gap > 0 else 0)

    denom = total_transitions + 1 + nb_gap
    entropy = 0.0
    for cell in cells:
        pi = (nb_gap / denom) if cell == "outside" else (user_cells.count(cell) / denom)
        if pi > 0:
            entropy -= pi * np.log2(pi)
    return entropy, n_states


def conditional_entropy(user_cells, user_stamps):
    """S_cond = H(next | current) from the user's own order-1 transition matrix,
    with the 'outside' state inserted on gaps > 4h10 (same alphabet as S_unc)."""
    n = len(user_cells)
    if n < 2:
        return 0.0

    trans = defaultdict(Counter)
    for i in range(n - 1):
        if user_stamps[i + 1] - user_stamps[i] <= GAP:
            trans[user_cells[i]][user_cells[i + 1]] += 1
        else:
            trans[user_cells[i]]["outside"] += 1
            trans["outside"][user_cells[i + 1]] += 1

    total = sum(sum(c.values()) for c in trans.values())
    if total == 0:
        return 0.0

    s_cond = 0.0
    for _, counter in trans.items():
        row_total = sum(counter.values())
        p_a = row_total / total
        h_a = 0.0
        for cnt in counter.values():
            p = cnt / row_total
            h_a -= p * np.log2(p)
        s_cond += p_a * h_a
    return s_cond


# ── Per-user pass ────────────────────────────────────────────────────────────

def hit_at_k(preds, true_next, k):
    for i in range(min(k, len(preds))):
        if preds[i][0] == true_next:
            return 1
    return 0


def run(test_users):
    import json
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    markov1 = NaiveMarkovChain(order=1, train_data=train_data)
    vomm = VOMM(max_order=MAX_CONTEXT, discount=VOMM_DISCOUNT,
                train_data=train_data,
                context_totals_train_data_path=CTX_TOTALS_PATH,
                unique_train_data_path=UNIQUE_PATH,
                unigram_train_data_path=UNIGRAM_PATH,
                DATASET_DIR=DATASET_DIR)

    records = []
    kept = 0
    for user in tqdm.tqdm(test_users, desc="Predictability vs accuracy"):
        n_cells = int(user[7])
        n_pred = n_cells - MAX_CONTEXT
        if n_pred < MIN_PREDICTIONS:
            continue

        cells = user[8::2]
        stamps = [int(ts) for ts in user[9::2]]

        v_c1 = v_c5 = m_c1 = m_c5 = 0
        for start in range(n_pred):
            window = cells[start:start + MAX_CONTEXT]
            true_next = cells[start + MAX_CONTEXT]
            last = window[-1]

            _, mk = markov1.predict_next(context=last, last=last, top_k=10)
            vm = vomm.predict_next(tuple(window))

            v_c1 += hit_at_k(vm, true_next, 1)
            v_c5 += hit_at_k(vm, true_next, 5)
            m_c1 += hit_at_k(mk, true_next, 1)
            m_c5 += hit_at_k(mk, true_next, 5)

        s_unc, n_states = uncorrelated_entropy(cells, stamps)
        s_cond = conditional_entropy(cells, stamps)

        records.append({
            "id_user": user[0],
            "n_pred": n_pred,
            "n_states": n_states,
            "vomm_acc1": v_c1 / n_pred,
            "vomm_acc5": v_c5 / n_pred,
            "markov_acc1": m_c1 / n_pred,
            "markov_acc5": m_c5 / n_pred,
            "S_unc": s_unc,
            "S_cond": s_cond,
            "pmax_unc": compute_pmax(s_unc, n_states),
            "pmax_cond": compute_pmax(s_cond, n_states),
        })

        kept += 1
        if kept >= N_USERS:
            break

    return pd.DataFrame(records)


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_scatter(df, xcol, xlabel, title, filename):
    sub = df.dropna(subset=[xcol, "vomm_acc1"])
    plt.figure(figsize=(7, 7))
    # One point per user. Small markers + low alpha so density still reads
    # despite the ~20k overlapping points.
    plt.scatter(sub[xcol], sub["vomm_acc1"], s=7, alpha=0.15,
                color="seagreen", edgecolors="none")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="y = x (model = ceiling)")
    plt.xlabel(xlabel)
    plt.ylabel("VOMM empirical ACC@1")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    out = PLOT_DIR / filename
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")
    plt.show()


def plot_bands(band_table):
    x = np.arange(len(band_table))
    width = 0.38
    plt.figure(figsize=(9, 6))
    b1 = plt.bar(x - width / 2, band_table["vomm_acc1"], width, label="VOMM", color="seagreen")
    b2 = plt.bar(x + width / 2, band_table["markov_acc1"], width, label="Markov order 1", color="steelblue")
    for bars in (b1, b2):
        plt.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
    for i, n in enumerate(band_table["n_users"]):
        plt.text(i, 0.02, f"n={n}", ha="center", fontsize=8, color="white")
    plt.xticks(x, [f"{lbl}\nP^max_cond {lo:.1f}-{hi:.1f}" for lo, hi, lbl in BANDS])
    plt.ylabel("Mean ACC@1 over users of the band")
    plt.ylim(0, 1)
    plt.title("Accuracy by predictability band")
    plt.legend()
    plt.tight_layout()
    out = PLOT_DIR / "predictability_by_band.png"
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")
    plt.show()


def plot_gap(df):
    gap = (df["pmax_cond"] - df["vomm_acc1"]).dropna()
    plt.figure(figsize=(10, 5))
    plt.hist(gap, bins=80, color="darkorange", edgecolor="white")
    plt.axvline(0, color="black", lw=1)
    plt.axvline(gap.mean(), color="red", ls="--", label=f"mean gap: {gap.mean():+.3f}")
    plt.xlabel("$P^{max}_{cond}$ - VOMM ACC@1  (predictability left on the table)")
    plt.ylabel("Number of users")
    plt.title("How far the VOMM stays below the order-1 ceiling")
    plt.legend()
    plt.tight_layout()
    out = PLOT_DIR / "predictability_gap.png"
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")
    plt.show()


if __name__ == "__main__":
    check_inputs()
    t0 = time.time()

    with open(TEST_PATH, "r", encoding="utf-8", newline="") as f:
        test_users = list(csv.reader(f, delimiter=";"))

    # Shuffle once (fixed seed) so the N_USERS sample spans all 15 days,
    # instead of only the first day(s) of the day-ordered test file.
    random.Random(SEED).shuffle(test_users)

    df = run(test_users)

    csv_out = OUTPUT_DIR / "predictability_vs_accuracy.csv"
    df.to_csv(csv_out, sep=";", index=False)

    # ── Stratified band table ────────────────────────────────────────────────
    rows = []
    for lo, hi, lbl in BANDS:
        band = df[(df["pmax_cond"] >= lo) & (df["pmax_cond"] < hi)]
        rows.append({"band": lbl, "n_users": len(band),
                     "vomm_acc1": band["vomm_acc1"].mean(),
                     "markov_acc1": band["markov_acc1"].mean(),
                     "pmax_cond": band["pmax_cond"].mean()})
    band_table = pd.DataFrame(rows)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\nUsers kept: {len(df)}   (>= {MIN_PREDICTIONS} predictions each)\n")
    print(f"Mean VOMM   ACC@1 : {df['vomm_acc1'].mean():.3f}")
    print(f"Mean Markov ACC@1 : {df['markov_acc1'].mean():.3f}")
    print(f"Mean P^max_unc    : {df['pmax_unc'].mean():.3f}"
          f"   | VOMM exceeds it for {(df['vomm_acc1'] > df['pmax_unc']).mean() * 100:.1f}% of users")
    print(f"Mean P^max_cond   : {df['pmax_cond'].mean():.3f}"
          f"   | VOMM exceeds it for {(df['vomm_acc1'] > df['pmax_cond']).mean() * 100:.1f}% of users")
    print(f"Corr ACC@1 / P^max_cond : {df['vomm_acc1'].corr(df['pmax_cond']):.3f}")
    print(f"Mean gap (P^max_cond - ACC@1) : {(df['pmax_cond'] - df['vomm_acc1']).mean():+.3f}")
    print("\nBy predictability band:")
    print(band_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nPer-user table -> {csv_out}")
    print(f"Total time: {time.time() - t0:.1f}s")

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_scatter(df, "pmax_cond", "Theoretical $P^{max}_{cond}$ (order-1 ceiling)",
                 "VOMM accuracy vs the order-1 predictability ceiling",
                 "predictability_vs_pmax_cond.png")
    plot_scatter(df, "pmax_unc", "Theoretical $P^{max}_{unc}$ (memoryless ceiling)",
                 "VOMM accuracy vs the memoryless ceiling (model should beat it)",
                 "predictability_vs_pmax_unc.png")
    plot_bands(band_table)
    plot_gap(df)
