# Sensitivity of the VOMM (order 5 + absolute-discounting back-off) to its discount d,
# on the no_duplicate test — the representation used for every result of the report —
# and on the same kind of users as PREDICTABILITY_vs_accuracy.py (test split shuffled
# with seed 67, users with at least 10 prediction points, 5-cell window).
#
# Replaces, for the report, the older study made with VOMM_V5 (temporal boosts) on the
# deduplicated test (VOMM_prediction_pipeline.py + PLOT_VOMM_metrics_by_discount.py).
#
# The model is loaded once; only `discount` changes between runs (it enters the
# probabilities only), and the prediction cache is cleared each time.
#
# Metrics: per-user mean ACC@k (report convention) and the mean log-loss in bits,
# -log2 P(true next cell), computed with the full back-off probability.
#
# Prerequisites: 1_train_test_split.py -> 2b_create_train_ngrams_matrix.py

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tqdm

from models import VOMM

MAIN_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = MAIN_DIR / "Database/no_duplicate"
MATRIX_DIR = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix"
TRAIN_DATA_PATH = MATRIX_DIR / "ngrams_matrix_train_random.json"
CTX_TOTALS_PATH = MATRIX_DIR / "context_totals_train.json"
UNIQUE_PATH = MATRIX_DIR / "unique_followers_train.json"
UNIGRAM_PATH = MATRIX_DIR / "unigram_train.json"
TEST_PATH = MAIN_DIR / "results/predictions/train_test/test_random.csv"

OUTPUT_DIR = MAIN_DIR / "results/predictions/simple_predictor/metrics/comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = MAIN_DIR / "results/predictions/simple_predictor/plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DISCOUNTS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
TOP_KS = [1, 3, 5, 10]
MAX_CONTEXT = 5
MIN_PREDICTIONS = 10
N_USERS = 5000             # a quarter of the 20,000 of the main table, for runtime
SEED = 67
EPS = 1e-12


def select_users():
    with open(TEST_PATH, "r", encoding="utf-8", newline="") as f:
        users = list(csv.reader(f, delimiter=";"))
    random.Random(SEED).shuffle(users)
    kept = [u for u in users if int(u[7]) - MAX_CONTEXT >= MIN_PREDICTIONS]
    return kept[:N_USERS]


def evaluate(vomm, users):
    acc = {k: [] for k in TOP_KS}
    losses = []
    for user in tqdm.tqdm(users, desc=f"d = {vomm.discount}"):
        cells = user[8::2]
        n_pred = int(user[7]) - MAX_CONTEXT
        hits = {k: 0 for k in TOP_KS}
        for start in range(n_pred):
            window = tuple(cells[start:start + MAX_CONTEXT])
            true_next = cells[start + MAX_CONTEXT]
            ranked = [c for c, _ in vomm.predict_next(window)[:max(TOP_KS)]]
            for k in TOP_KS:
                hits[k] += int(true_next in ranked[:k])
            losses.append(-math.log2(max(vomm.prob(window, true_next), EPS)))
        for k in TOP_KS:
            acc[k].append(hits[k] / n_pred)
    res = {f"ACC@{k}": float(np.mean(acc[k])) for k in TOP_KS}
    res["log_loss_bits"] = float(np.mean(losses))
    res["n_predictions"] = len(losses)
    return res


def plot(results):
    ds = [r["discount"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for k, marker in zip(TOP_KS, "osd^"):
        axes[0].plot(ds, [r[f"ACC@{k}"] for r in results], marker=marker, label=f"ACC@{k}")
    axes[0].set_xlabel("Discount $d$")
    axes[0].set_ylabel("Mean accuracy over users")
    axes[0].set_title("Accuracy as a function of the discount")
    axes[0].legend()
    axes[1].plot(ds, [r["log_loss_bits"] for r in results], "o--", color="purple")
    axes[1].set_xlabel("Discount $d$")
    axes[1].set_ylabel("Mean log-loss (bits)")
    axes[1].set_title("Log-loss as a function of the discount")
    for a in axes:
        a.grid(ls="--", alpha=0.5)
        a.set_xticks(ds)
    fig.suptitle(f"VOMM (order 5 + back-off), no_duplicate test, {len(users):,} users, "
                 f"{results[0]['n_predictions']:,} predictions")
    plt.tight_layout()
    out = PLOT_DIR / "vomm_discount_no_duplicate.png"
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    t0 = time.time()
    users = select_users()
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    vomm = VOMM(max_order=MAX_CONTEXT, discount=DISCOUNTS[0], train_data=train_data,
                context_totals_train_data_path=CTX_TOTALS_PATH, unique_train_data_path=UNIQUE_PATH,
                unigram_train_data_path=UNIGRAM_PATH, DATASET_DIR=DATASET_DIR)
    train_data = None

    results = []
    for d in DISCOUNTS:
        vomm.discount = d
        vomm.cache = {}
        r = evaluate(vomm, users)
        r["discount"] = d
        results.append(r)
        print(json.dumps(r))

    with open(OUTPUT_DIR / "vomm_discount_no_duplicate.json", "w", encoding="utf-8") as f:
        json.dump({"n_users": len(users), "results": results}, f, indent=2)
    plot(results)
    print(f"Total time: {time.time() - t0:.1f}s")
