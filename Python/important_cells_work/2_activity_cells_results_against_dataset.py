from pathlib import Path
import pandas as pd
import ast
import csv

MAIN_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = MAIN_DIR / "Database/no_duplicate"
INTERMEDIATE_RESULT = MAIN_DIR / "results/intermediate_result"
INTERMEDIATE_RESULT.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = INTERMEDIATE_RESULT / "classified_dataset_merge_no_merge.csv"
OUTPUT_PATH  = INTERMEDIATE_RESULT / "comparison_vs_dataset.csv"

# Only keep the canonical period to have one row per (user, day)
PERIOD = "Home_04h10-19h50_Activity_05h00-19h00"


# === 1. Reference activity cell from the dataset (column BS2, index 6) ===
# Format: user_id;age;gender;unknown;letters;BS1(home);BS2(activity);n_records;cell;ts;...
ref_cells: dict[tuple[str, str], str] = {}

for file in INPUT_DIR.glob("*.csv"):
    day = file.name.split("_")[0]
    with open(file, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            if len(line) < 7:
                continue
            ref_cells[(line[0], day)] = line[6]  # BS2


def parse_cells(s: str) -> list[str]:
    if pd.isna(s) or s in ('', 'None', '[]'):
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


# === 2. Load classified results (canonical period only) ===
df = pd.read_csv(RESULTS_PATH, sep=";",
                 dtype={"user_id": str, "day": str, "activity_cells": str,
                        "reason_None": str, "period": str,
                        "working_period": str, "nb_activity_cells": str})

df = df[df["period"] == PERIOD].copy()

df["algo_cells"] = df["activity_cells"].apply(parse_cells)
df["ref_cell"]   = df.apply(lambda r: ref_cells.get((r["user_id"], r["day"]), ""), axis=1)

df["algo_has"] = df["algo_cells"].apply(lambda c: len(c) > 0)
df["ref_has"]  = df["ref_cell"].ne("")


# === 3. Classify each row ===
def classify(row):
    ref   = row["ref_has"]
    algo  = row["algo_has"]
    match = ref and (row["ref_cell"] in row["algo_cells"])

    if match:
        return "match"
    elif ref and algo:
        return "different"    # both have a cell but they differ
    elif ref and not algo:
        return "only_dataset" # dataset has a cell, algo found none
    elif not ref and algo:
        return "only_algo"    # algo found a cell, dataset has none
    else:
        return "neither"      # neither side has a cell


df["case"] = df.apply(classify, axis=1)


# === 4. Day-by-day table ===
result = (
    df.groupby(["day", "case"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["match", "different", "only_dataset", "only_algo", "neither"], fill_value=0)
    .reset_index()
    .rename(columns={
        "day":          "day",
        "match":        "match",
        "different":    "different_cell",
        "only_dataset": "dataset_only",
        "only_algo":    "algo_only",
        "neither":      "no_cell",
    })
)

result.to_csv(OUTPUT_PATH, sep=";", index=False)
print(result.to_string(index=False))


# === 5. One example (user_id, day) per case ===
CASES = ["match", "different", "only_dataset", "only_algo", "neither"]
print("\n--- One example per case ---")
for case in CASES:
    row = df[df["case"] == case].iloc[0] if not df[df["case"] == case].empty else None
    if row is None:
        print(f"{case:15s} : no example found")
    else:
        print(f"{case:15s} : user={row['user_id']}  day={row['day']}"
              f"  algo={row['algo_cells']}  dataset_ref='{row['ref_cell']}'")
