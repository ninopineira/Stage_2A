from pathlib import Path
from typing import Callable
import csv
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

MAIN_DIR            = Path(__file__).parent.parent.parent
INPUT_DIR           = MAIN_DIR / "Database/no_duplicate"
INTERMEDIATE_RESULT = MAIN_DIR / "results/intermediate_result"
INTERMEDIATE_RESULT.mkdir(parents=True, exist_ok=True)

MERGE = "no_merge"

OUTPUT_DIR = INTERMEDIATE_RESULT / "home_cells_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)

PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# ── Home cell detection (same logic as in 2_get_user_act_cell_continue.py) ────
def get_home_cell(cells: list[str], stamps: list[int],
                  merge_func: Callable[[str], str] = lambda x: x) -> str | None:
    MORNING_END   = 4 * 3600 + 10 * 60   # 04:10:00 = 15 000 s
    EVENING_START = 19 * 3600 + 50 * 60  # 19:50:00 = 71 400 s

    if len(cells) < 2:
        return None

    cells_m = [merge_func(c) for c in cells]
    

    morning_cells = {c for c, t in zip(cells_m, stamps) if t <= MORNING_END}
    evening_cells = {c for c, t in zip(cells_m, stamps) if t >= EVENING_START}

    candidates = morning_cells & evening_cells
    if not candidates:
        return None

    for c in cells_m:
        if c in candidates:
            return c

    return None


# ── 1. Reference home cells from the dataset (column BS1, index 5) ────────────
ref_homes: dict[tuple[str, str], str] = {}
for file in INPUT_DIR.glob("*.csv"):
    day = file.name.split("_")[0]
    with open(file, mode="r", encoding="utf-8", newline="") as f:
        for line in csv.reader(f, delimiter=";"):
            if len(line) >= 6:
                ref_homes[(line[0], day)] = line[5]


# ── 2. Compute algo home cell for every user×day ──────────────────────────────
rows = []
files = sorted(INPUT_DIR.glob("*.csv"))

for file in tqdm.tqdm(files, desc="Computing home cells"):
    day = file.name.split("_")[0]
    with open(file, mode="r", encoding="utf-8", newline="") as f:
        for line in csv.reader(f, delimiter=";"):
            if not line:
                continue
            user_id     = line[0]
            user_cells  = [c  for c  in line[8::2] if c]
            user_stamps = [int(ts) for ts in line[9::2] if ts]

            algo_home = get_home_cell(user_cells, user_stamps)
            ref_home  = ref_homes.get((user_id, day), "")

            rows.append({
                "user_id":   user_id,
                "day":       day,
                "algo_home": algo_home,
                "ref_home":  ref_home,
            })

df = pd.DataFrame(rows)
df["ref_has"]  = df["ref_home"].ne("")
df["algo_has"] = df["algo_home"].notna()


# ── 3. Classify each user×day ─────────────────────────────────────────────────
def classify(row):
    match = row["ref_has"] and (row["ref_home"] == row["algo_home"])
    if match:                               return "match"
    if row["ref_has"] and row["algo_has"]:  return "different"
    if row["ref_has"]:                      return "dataset_only"
    if row["algo_has"]:                     return "algo_only"
    return "neither"

df["case"] = df.apply(classify, axis=1)

counts = (
    df.groupby(["day", "case"]).size().unstack(fill_value=0)
      .reindex(columns=["match", "different", "dataset_only", "algo_only"], fill_value=0)
      .reset_index()
)
totals = df.groupby("day").agg(
    total_with_ref  =("ref_has",  "sum"),
    total_with_algo =("algo_has", "sum"),
    total_users     =("user_id",  "count"),
).reset_index()

table = counts.merge(totals, on="day")


# ── 4. Print results ──────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print(f"  HOME CELL COMPARISON VS DATASET  (merge={MERGE})")
print(f"{'═'*70}")
print(table.to_string(index=False))

print("\nExamples (one per case):")
for case in ["match", "different", "dataset_only", "algo_only"]:
    sub = df[df["case"] == case]
    if sub.empty:
        print(f"  {case:15s}: no example")
    else:
        r = sub.iloc[0]
        print(f"  {case:15s}: user={r['user_id']}  day={r['day']}"
              f"  algo='{r['algo_home']}'  ref='{r['ref_home']}'")


# ── 5. Save CSV ───────────────────────────────────────────────────────────────
out_path = OUTPUT_DIR / "home_cells_vs_dataset.csv"
table.to_csv(out_path, sep=";", index=False)
print(f"\nSaved: {out_path}")


# ── 6. Plots ──────────────────────────────────────────────────────────────────
CASE_COLORS = {
    "match":        "#55A868",
    "different":    "#DD8452",
    "dataset_only": "#4C72B0",
    "algo_only":    "#C44E52",
}
CASE_LABELS = {
    "match":        "Match",
    "different":    "Different cell",
    "dataset_only": "Dataset only",
    "algo_only":    "Algo only",
}

# Plot 1 — Match rate per day
fig, ax = plt.subplots(figsize=(14, 5))
ref = table["total_with_ref"].replace(0, np.nan)
ax.plot(table["day"], table["match"] / ref,
        marker="o", markersize=6, linewidth=2, color="#4C72B0")
ax.set_title("Home cell match rate vs dataset per day  (no_merge)", fontweight="bold")
ax.set_xlabel("Day")
ax.set_ylabel("Match rate  (match / users with ref home cell)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_home_match_rate_by_day.png", dpi=150)
plt.close()

# Plot 2 — Stacked breakdown per day (normalised to total users)
fig, ax = plt.subplots(figsize=(14, 5))
x      = np.arange(len(table))
total  = table["total_users"].replace(0, np.nan)
bottom = np.zeros(len(table))
for case in ["match", "different", "dataset_only", "algo_only"]:
    vals = (table[case] / total).fillna(0).to_numpy()
    ax.bar(x, vals, bottom=bottom, color=CASE_COLORS[case],
           label=CASE_LABELS[case], width=0.75, edgecolor="white", linewidth=0.8)
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels(table["day"].tolist(), rotation=45, fontsize=7)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_title("Home cell breakdown per day  (% of total users, no_merge)", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_home_breakdown_per_day.png", dpi=150)
plt.close()

# Plot 3 — Coverage per day
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(table["day"], table["total_users"],
        color="#333", linewidth=1.5, linestyle=":", label="Total users")
ax.plot(table["day"], table["total_with_ref"],
        color="#888", linewidth=2, linestyle="--", label="Users with ref home cell (dataset)")
ax.plot(table["day"], table["total_with_algo"],
        marker="o", markersize=5, linewidth=2, color="#4C72B0",
        label="Users with algo home cell")
ax.set_title("Home cell coverage per day  (no_merge)", fontweight="bold")
ax.set_xlabel("Day")
ax.set_ylabel("Number of users")
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_home_coverage_by_day.png", dpi=150)
plt.close()

print(f"Plots saved in: {PLOT_DIR}")
