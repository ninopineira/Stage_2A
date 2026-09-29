# Compares an order-1 Markov chain (the core of the Machin_learning/ approach) to
# the VOMM (variable-order Markov + absolute-discounting back-off) on EXACTLY the
# same DEDUPLICATED test set and the same metrics, so the numbers are directly comparable.
# (The same comparison on the no_duplicate test, with the "stay" baseline, is made by
# PREDICTABILITY_vs_accuracy.py -> markov_vs_vomm_no_duplicate.png.)
#
# Why this and not the Machin_learning/ numbers directly:
#   - Machin_learning/markov_sequential_prediction.py runs per-user on
#     Database/no_duplicate, which keeps consecutive self-transitions (72% of
#     records) -> its accuracy is inflated by the trivial "stay" case and is not
#     comparable to the VOMM numbers.
#   - Here both models are trained on the SAME train n-grams and evaluated on the
#     SAME deduplicated global test split, at the SAME prediction points, with the
#     SAME ACC@k / MAP@k. Only the model changes. That isolates what the higher
#     orders + backoff of the VOMM actually buy over a plain order-1 Markov.
#
# Prerequisites (dataset_creation pipeline):
#   1_train_test_split.py  ->  3_deduplicate_csv.py  ->  2b_create_train_ngrams_matrix.py

import json
import csv
import time
from pathlib import Path

import tqdm
import matplotlib.pyplot as plt

from models import NaiveMarkovChain, VOMM
from utils import Metrics

MAIN_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = MAIN_DIR / "Database/no_duplicate"

TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/ngrams_matrix_train_random.json"
TEST_PATH = MAIN_DIR / "results/predictions/train_test/removed_repeat/test_random.csv"
CTX_TOTALS_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/context_totals_train.json"
UNIQUE_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/unique_followers_train.json"
UNIGRAM_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/unigram_train.json"

OUTPUT_DIR = MAIN_DIR / "results/predictions/simple_predictor/metrics/comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = MAIN_DIR / "results/predictions/simple_predictor/plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTEXT = 5          # VOMM order; the order-1 model uses only the last cell of the same window
TOP_KS = [1, 3, 5, 10]
N_USERS = 20000           # cap for a reasonable runtime (same order of magnitude as the VOMM pipeline)
VOMM_DISCOUNT = 0.9


def check_inputs():
    for p in (TRAIN_DATA_PATH, TEST_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.\nRun the dataset_creation pipeline first: "
                "1_train_test_split.py -> 3_deduplicate_csv.py -> 2b_create_train_ngrams_matrix.py"
            )


def collect_predictions(test_users):
    """Run both models over the same prediction points and return, for each
    model, the list of ranked predictions plus the shared list of true next cells."""
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    markov1 = NaiveMarkovChain(order=1, train_data=train_data)
    vomm = VOMM(max_order=MAX_CONTEXT, discount=VOMM_DISCOUNT,
                train_data=train_data,
                context_totals_train_data_path=CTX_TOTALS_PATH,
                unique_train_data_path=UNIQUE_PATH,
                unigram_train_data_path=UNIGRAM_PATH,
                DATASET_DIR=DATASET_DIR)

    true_next_list = []
    markov1_preds = []
    vomm_preds = []

    seen = 0
    for user in tqdm.tqdm(test_users, desc="Comparing Markov(1) vs VOMM"):
        n_cells = int(user[7])
        if n_cells < MAX_CONTEXT + 1:
            continue
        cells = user[8::2]
        for start in range(n_cells - MAX_CONTEXT):
            window = cells[start:start + MAX_CONTEXT]
            true_next = cells[start + MAX_CONTEXT]
            last = window[-1]

            _, mk_full = markov1.predict_next(context=last, last=last, top_k=10)
            vm_full = vomm.predict_next(tuple(window))

            true_next_list.append(true_next)
            markov1_preds.append(mk_full)
            vomm_preds.append(vm_full)

        seen += 1
        if seen >= N_USERS:
            break

    return true_next_list, markov1_preds, vomm_preds, seen


def score(metric_man, preds_list, true_next_list):
    res = {}
    for k in TOP_KS:
        res[f"ACC@{k}"] = metric_man.top_k_accuracy(preds_list=preds_list, true_next=true_next_list, k=k)
        res[f"MAP@{k}"] = metric_man.map_k(preds_list=preds_list, true_next=true_next_list, k=k)
    res["log-l_mean"] = metric_man.log_likelihood(preds_list=preds_list, true_next_list=true_next_list)
    return res


def plot_comparison(results):
    x = range(len(TOP_KS))
    width = 0.38
    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], [results["Markov order 1"][f"ACC@{k}"] for k in TOP_KS],
            width, label="Markov order 1", color="steelblue")
    plt.bar([i + width / 2 for i in x], [results["VOMM (order 5 + backoff)"][f"ACC@{k}"] for k in TOP_KS],
            width, label="VOMM (order 5 + backoff)", color="seagreen")
    for i, k in enumerate(TOP_KS):
        plt.text(i - width / 2, results["Markov order 1"][f"ACC@{k}"] + 0.005,
                 f"{results['Markov order 1'][f'ACC@{k}']:.3f}", ha="center", fontsize=8)
        plt.text(i + width / 2, results["VOMM (order 5 + backoff)"][f"ACC@{k}"] + 0.005,
                 f"{results['VOMM (order 5 + backoff)'][f'ACC@{k}']:.3f}", ha="center", fontsize=8)
    plt.xticks(list(x), [f"ACC@{k}" for k in TOP_KS])
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Next-cell prediction: order-1 Markov vs VOMM\n(same deduplicated test set, same metric)")
    plt.legend()
    plt.tight_layout()
    out = PLOT_DIR / "markov_vs_vomm_dedup.png"
    plt.savefig(out, dpi=150)
    print(f"Saved plot -> {out}")
    plt.show()


if __name__ == "__main__":
    check_inputs()
    t0 = time.time()

    with open(TEST_PATH, "r", encoding="utf-8", newline="") as f:
        test_users = list(csv.reader(f, delimiter=";"))

    true_next_list, markov1_preds, vomm_preds, n_seen = collect_predictions(test_users)

    metric_man = Metrics()
    results = {
        "Markov order 1": score(metric_man, markov1_preds, true_next_list),
        "VOMM (order 5 + backoff)": score(metric_man, vomm_preds, true_next_list),
    }
    results["meta"] = {"n_users": n_seen, "n_predictions": len(true_next_list),
                       "max_context": MAX_CONTEXT, "vomm_discount": VOMM_DISCOUNT}

    out_json = OUTPUT_DIR / "markov_vs_vomm_dedup.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ── Console table ────────────────────────────────────────────────────────
    print(f"\nUsers evaluated: {n_seen}   |   predictions: {len(true_next_list):,}\n")
    header = f"{'metric':<12}{'Markov order 1':>18}{'VOMM':>12}{'gain':>10}"
    print(header)
    print("-" * len(header))
    for k in TOP_KS:
        m = results["Markov order 1"][f"ACC@{k}"]
        v = results["VOMM (order 5 + backoff)"][f"ACC@{k}"]
        print(f"{'ACC@'+str(k):<12}{m:>18.3f}{v:>12.3f}{v - m:>+10.3f}")
    print(f"\nMetrics saved -> {out_json}")
    print(f"Total time: {time.time() - t0:.1f}s")

    plot_comparison(results)
