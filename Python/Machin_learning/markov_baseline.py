import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import tqdm

"""
Personal first-order Markov baseline.

For each user (each user only exists on a single day, ids are not stable
across days), we split their transitions cell_from -> cell_to into a random
70% train / 30% test set (random, not chronological, so that both sets cover
a mix of hours and the model isn't unfairly tested on a time-of-day behaviour
it never saw in train). We build the user's own transition-count matrix from
the train transitions, predict each test transition's cell_to by argmax over
the train counts for that cell_from (falling back to the user's single most
frequent cell when cell_from was never seen in train), and record the
resulting accuracy. This is meant to be compared against the theoretical
P^max computed in maximal_previsibility.py for the same users.
"""

MAIN_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = MAIN_DIR / "Database/no_duplicate"
OUTPUT_DIR = MAIN_DIR / "results/numpy"

GAP_LIMIT = 4 * 3600 + 600  # same "outside" threshold used across the project
MIN_RECORDS = 10  # same threshold as sample_for_training.py
TRAIN_FRAC = 0.7
RANDOM_SEED = 0

files = sorted(INPUT_DIR.glob("*.csv"))


def get_day(filepath: Path) -> str:
    return filepath.name.split("_")[0]


def build_transitions(user_cells, user_stamps):
    """Return the list of (cell_from, cell_to) transitions, inserting the
    'outside' state on both sides of a gap > 4h10 (same convention as
    add_an_user_s_transitions / entropy_cells_by_user)."""
    transitions = []
    for i in range(len(user_cells) - 1):
        if user_stamps[i + 1] - user_stamps[i] <= GAP_LIMIT:
            transitions.append((user_cells[i], user_cells[i + 1]))
        else:
            transitions.append((user_cells[i], "outside"))
            transitions.append(("outside", user_cells[i + 1]))
    return transitions


def predict_accuracy(train_transitions, test_transitions, rng):
    """Build a personal transition-count model from train_transitions and
    evaluate next-cell prediction accuracy on test_transitions."""
    counts = {}
    for cell_from, cell_to in train_transitions:
        counts.setdefault(cell_from, Counter())[cell_to] += 1

    # Fallback for cold-start cell_from: the user's single most common cell
    # (either side of a transition) seen in train.
    overall_counts = Counter()
    for cell_from, cell_to in train_transitions:
        overall_counts[cell_from] += 1
        overall_counts[cell_to] += 1
    fallback_cell = overall_counts.most_common(1)[0][0] if overall_counts else None

    correct = 0
    for cell_from, cell_to in test_transitions:
        if cell_from in counts:
            best_count = max(counts[cell_from].values())
            candidates = [c for c, n in counts[cell_from].items() if n == best_count]
            prediction = rng.choice(candidates)
        else:
            prediction = fallback_cell

        if prediction == cell_to:
            correct += 1

    return correct / len(test_transitions)


def run(train_frac=TRAIN_FRAC, min_records=MIN_RECORDS, seed=RANDOM_SEED):
    rng = random.Random(seed)
    results = {}

    for file in tqdm.tqdm(files, desc="Markov baseline"):
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

                shuffled = transitions[:]
                rng.shuffle(shuffled)
                n_train = max(1, round(train_frac * len(shuffled)))
                n_train = min(n_train, len(shuffled) - 1)  # keep at least 1 test transition
                train_transitions = shuffled[:n_train]
                test_transitions = shuffled[n_train:]

                accuracy = predict_accuracy(train_transitions, test_transitions, rng)

                results[id_user] = (day, accuracy, len(train_transitions),
                                     len(test_transitions), n_records)

    np.save(OUTPUT_DIR / "markov_accuracy_no_merge.npy", results)
    return results


if __name__ == "__main__":
    from maximal_previsibility import load_pmax_dataframe

    results = run()

    # Zero-entropy users (never left a single cell) are excluded from the
    # reported figures: they are trivially predictable and would inflate the
    # mean. load_pmax_dataframe() already drops them.
    kept = set(load_pmax_dataframe()['id_user'])
    accuracies = [v[1] for u, v in results.items() if u in kept]

    print(f"Users predicted        : {len(results)}")
    print(f"Users kept (entropy>0) : {len(accuracies)}")
    print(f"Mean accuracy          : {np.mean(accuracies):.3f}")
    print(f"Median accuracy        : {np.median(accuracies):.3f}")
