import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import tqdm
import matplotlib.pyplot as plt

from markov_baseline import GAP_LIMIT, MIN_RECORDS, build_transitions, get_day
from maximal_previsibility import load_pmax_dataframe

"""
Causal / online next-cell prediction: this is the actual goal of the project
(predict the next station knowing only what happened before), as opposed to
markov_baseline.py's random 70/30 split which only answers "do I capture my
own P^max on an arbitrary sample of my trips".

For each user (single day), we walk through their transitions in chronological
order. At transition i we predict cell_to using only the transition counts
accumulated from transitions 0..i-1 (the strict past); we then add transition
i to the counts before moving on (the personal model grows online, one
observation at a time). The very first transition of the day cannot be
predicted (no history yet) and is excluded from the accuracy.
"""

MAIN_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = MAIN_DIR / "Database/no_duplicate"
OUTPUT_DIR = MAIN_DIR / "results/numpy"

RANDOM_SEED = 0

files = sorted(INPUT_DIR.glob("*.csv"))


def sequential_predictions(transitions, rng):
    """Walk through the transitions in chronological order and yield
    (index, cell_from, cell_to, prediction) for each one that can be predicted
    from the strict past. The model is updated online: transition i is added to
    the counts only after it has been predicted. Transition 0 has no history and
    is therefore never yielded."""
    counts = {}
    overall_counts = Counter()

    for i, (cell_from, cell_to) in enumerate(transitions):
        if i > 0:
            if cell_from in counts:
                best_count = max(counts[cell_from].values())
                candidates = [c for c, n in counts[cell_from].items() if n == best_count]
                prediction = rng.choice(candidates)
            else:
                prediction = overall_counts.most_common(1)[0][0]

            yield i, cell_from, cell_to, prediction

        counts.setdefault(cell_from, Counter())[cell_to] += 1
        overall_counts[cell_from] += 1
        overall_counts[cell_to] += 1


def sequential_accuracy(transitions, rng):
    """Accuracy of the online next-cell prediction. Returns None if there is
    no evaluable transition (i.e. fewer than 2 transitions total)."""
    correct = 0
    total_predicted = 0

    for _, _, cell_to, prediction in sequential_predictions(transitions, rng):
        total_predicted += 1
        if prediction == cell_to:
            correct += 1

    if total_predicted == 0:
        return None
    return correct / total_predicted


def run(min_records=MIN_RECORDS, seed=RANDOM_SEED):
    rng = random.Random(seed)
    results = {}

    for file in tqdm.tqdm(files, desc="Sequential Markov prediction"):
        day = get_day(file)
        with open(file, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            for line in reader:
                id_user = line[0]
                user_cells = [c for c in line[8::2] if c]
                user_stamps = [int(ts) for ts in line[9::2] if ts]

                n_records = len(user_cells)
                if n_records < min_records:
                    continue

                transitions = build_transitions(user_cells, user_stamps)
                if len(transitions) < 2:
                    continue

                accuracy = sequential_accuracy(transitions, rng)
                if accuracy is None:
                    continue

                results[id_user] = (day, accuracy, len(transitions), n_records)

    np.save(OUTPUT_DIR / "markov_sequential_accuracy_no_merge.npy", results)
    return results


if __name__ == "__main__":
    results = run()

    # Zero-entropy users (never left a single cell) are excluded: they are not
    # in load_pmax_dataframe(), and keeping them would inflate the accuracy with
    # users who are trivially predictable.
    pmax_df = load_pmax_dataframe()
    pmax_by_user = dict(zip(pmax_df['id_user'], pmax_df['pmax']))
    kept = [u for u in results if u in pmax_by_user]

    accuracies = np.array([results[u][1] for u in kept])
    pmax_values = np.array([pmax_by_user[u] for u in kept])

    print(f"Users predicted        : {len(results)}")
    print(f"Users kept (entropy>0) : {len(kept)}")
    print(f"Mean accuracy          : {accuracies.mean():.3f}")
    print(f"Median accuracy        : {np.median(accuracies):.3f}")
    print(f"Mean P^max             : {pmax_values.mean():.3f}")

    # ── Compare against the random-split baseline, if available ───────────────
    batch_path = OUTPUT_DIR / "markov_accuracy_no_merge.npy"
    if batch_path.exists():
        batch_raw = np.load(batch_path, allow_pickle=True).item()
        batch_accuracies = np.array([batch_raw[u][1] for u in kept if u in batch_raw])
        print(f"\nFor the same users, random-split (markov_baseline.py) mean accuracy: "
              f"{batch_accuracies.mean():.3f}")

    plt.figure(figsize=(10, 5))
    plt.hist(accuracies, bins=100, alpha=0.6, label=f"Sequential accuracy (mean {accuracies.mean():.2f})",
             color='seagreen')
    if batch_path.exists():
        plt.hist(batch_accuracies, bins=100, alpha=0.4,
                  label=f"Random-split accuracy (mean {batch_accuracies.mean():.2f})", color='steelblue')
    plt.hist(pmax_values, bins=100, alpha=0.3, label=f"$P^{{max}}$ (mean {pmax_values.mean():.2f})", color='orange')
    plt.xlabel("Predictability")
    plt.ylabel("Number of Users")
    plt.title("Sequential (causal) accuracy vs. random-split accuracy vs. $P^{max}$")
    plt.legend()
    plt.tight_layout()
    plt.show()
