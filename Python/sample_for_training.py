import csv
from pathlib import Path

import numpy as np
import tqdm

"""
Builds a training sample from the first 4 days of the dataset.

For each of the 4 days, we keep the 100 users with the highest S_unc entropy
(same definition as Machin_learning/transition_emtropy.py::entropy_for_user)
among users who have between 10 and 200 records that day and no gap greater
than 4h between two consecutive records.

The selected lines are copied as-is (same columns, same order) into one CSV
per day.
"""

MAIN_DIR   = Path(__file__).parent.parent
INPUT_DIR  = MAIN_DIR / "Database/no_duplicate"
OUTPUT_DIR = MAIN_DIR / "Database/sample_for_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_DAYS       = 15
N_USERS      = 2000
MIN_RECORDS  = 10
MAX_RECORDS  = 200
GAP_LIMIT    = 4 * 3600 + 60  # 4h (+60s buffer, same threshold used elsewhere in the project)


def get_day(filepath: Path) -> str:
    return filepath.name.split("_")[0]


def entropy_for_user(user_cells: list[str]) -> float:
    """S_unc: uncorrelated Shannon entropy of the cells visited by the user."""
    total_records = len(user_cells)
    if total_records == 0:
        return 0.0

    counts = {}
    for cell in user_cells:
        counts[cell] = counts.get(cell, 0) + 1

    entropy_user = 0.0
    for count in counts.values():
        pi = count / total_records
        if pi > 0:
            entropy_user -= pi * np.log2(pi)

    return entropy_user


def has_no_large_gap(user_stamps: list[int]) -> bool:
    return all(
        user_stamps[i + 1] - user_stamps[i] <= GAP_LIMIT
        for i in range(len(user_stamps) - 1)
    )


def select_top_users_for_day(file: Path) -> list[list[str]]:
    """Return the raw CSV rows of the N_USERS users with the highest entropy for that day."""
    candidates = []  # (entropy, raw_line)

    with open(file, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for line in reader:
            user_cells = [c for c in line[8::2] if c]
            user_stamps = [int(ts) for ts in line[9::2] if ts]

            n_records = len(user_cells)
            if not (MIN_RECORDS <= n_records <= MAX_RECORDS):
                continue
            if not has_no_large_gap(user_stamps):
                continue

            entropy_user = entropy_for_user(user_cells)
            candidates.append((entropy_user, line))

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [line for _, line in candidates[:N_USERS]]


if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob("*.csv"))[:N_DAYS]

    for file in tqdm.tqdm(files, desc="Sampling days"):
        day = get_day(file)
        top_lines = select_top_users_for_day(file)

        out_path = OUTPUT_DIR / f"{day}_sample_for_training.csv"
        with open(out_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerows(top_lines)

        print(f"{day}: {len(top_lines)} users selected -> {out_path}")
