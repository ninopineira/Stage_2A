from pathlib import Path
from itertools import combinations
import pandas as pd
import ast
import csv


MAIN_DIR            = Path(__file__).parent.parent.parent
INPUT_DIR           = MAIN_DIR / "Database/no_duplicate"
INTERMEDIATE_RESULT = MAIN_DIR / "results/intermediate_result"
INTERMEDIATE_RESULT.mkdir(parents=True, exist_ok=True)

PERIOD  = "Home_04h10-19h50_Activity_05h00-19h00"
MERGE   = "no_merge"
METHODS = ["cont_no_gap", "cont_gap", "no_cont_no_gap", "no_cont_gap"]

OUTPUT_DIR = INTERMEDIATE_RESULT / "activity_cells_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── 1. Reference cells from the dataset (BS1=home index 5, BS2=activity index 6)
ref_cells: dict[tuple[str, str], str] = {}
ref_homes: dict[tuple[str, str], str] = {}
for file in INPUT_DIR.glob("*.csv"):
    day = file.name.split("_")[0]
    with open(file, mode="r", encoding="utf-8", newline="") as f:
        for line in csv.reader(f, delimiter=";"):
            if len(line) >= 7:
                ref_cells[(line[0], day)] = line[6]
                ref_homes[(line[0], day)] = line[5]


def parse_cells(s) -> list[str]:
    if pd.isna(s) or s in ("", "None", "[]"):
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def parse_num_list(s) -> list[float]:
    if pd.isna(s) or s in ("", "None", "[]"):
        return []
    try:
        return [float(v) for v in ast.literal_eval(s)]
    except Exception:
        return []


def load_method(method: str) -> pd.DataFrame | None:
    path = INTERMEDIATE_RESULT / f"classified_dataset_{method}_merge_{MERGE}.csv"
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return None
    df = pd.read_csv(path, sep=";",
                     dtype={"user_id": str, "day": str, "activity_cells": str,
                            "reason_None": str, "period": str,
                            "working_period": str, "nb_activity_cells": str,
                            "cell_times": str, "cell_records": str})
    df = df[df["period"] == PERIOD].copy()
    df["algo_cells"]   = df["activity_cells"].apply(parse_cells)
    df["cell_times"]   = df["cell_times"].apply(parse_num_list)
    df["cell_records"] = df["cell_records"].apply(parse_num_list)
    df["algo_first"]   = df["algo_cells"].apply(lambda x: x[0] if x else None)
    df["algo_has"]     = df["algo_cells"].apply(bool)

    # Method A : first cell in list (already algo_first)
    df["algo_A"] = df["algo_first"]

    # Method B : cell with most time
    def select_B(r):
        if not r["cell_times"]:
            return None
        idx = r["cell_times"].index(max(r["cell_times"]))
        return r["algo_cells"][idx] if idx < len(r["algo_cells"]) else None

    # Method C : cell with most records
    def select_C(r):
        if not r["cell_records"]:
            return None
        idx = r["cell_records"].index(max(r["cell_records"]))
        return r["algo_cells"][idx] if idx < len(r["algo_cells"]) else None

    df["algo_B"] = df.apply(select_B, axis=1)
    df["algo_C"] = df.apply(select_C, axis=1)
    return df


# ── 2. Vs-dataset table with 3 selection methods (A / B / C) ─────────────────
def build_vs_dataset(df: pd.DataFrame):
    df = df.copy()
    df["ref_cell"]     = df.apply(lambda r: ref_cells.get((r["user_id"], r["day"]), ""), axis=1)
    df["ref_has"]      = df["ref_cell"].ne("")
    df["ref_home_has"] = df.apply(lambda r: bool(ref_homes.get((r["user_id"], r["day"]), "")), axis=1)

    def classify(ref_has, ref_cell, algo_sel, algo_has):
        match = ref_has and (ref_cell == algo_sel)
        if match:                        return "match"
        if ref_has and algo_has:         return "different"
        if ref_has:                      return "dataset_only"
        if algo_has:                     return "algo_only"
        return "neither"

    for s in ["A", "B", "C"]:
        df[f"case_{s}"] = df.apply(
            lambda r, s=s: classify(r["ref_has"], r["ref_cell"], r[f"algo_{s}"], r["algo_has"]),
            axis=1
        )

    # Aggregate per day per strategy
    result = None
    for s in ["A", "B", "C"]:
        cnt = (df.groupby(["day", f"case_{s}"]).size()
               .unstack(fill_value=0)
               .reindex(columns=["match", "different", "dataset_only", "algo_only"], fill_value=0)
               .rename(columns=lambda c: f"{c}_{s}")
               .reset_index())
        result = cnt if result is None else result.merge(cnt, on="day")

    # dataset_only / algo_only don't depend on the selection strategy — keep only _A
    result = result.drop(
        columns=[f"dataset_only_{s}" for s in ["B", "C"]] +
                [f"algo_only_{s}" for s in ["B", "C"]]
    )
    result = result.rename(columns={"dataset_only_A": "dataset_only",
                                    "algo_only_A":    "algo_only"})

    totals = df.groupby("day").agg(
        total_with_ref  =("ref_has",      "sum"),
        total_with_algo =("algo_has",     "sum"),
        total_with_home =("ref_home_has", "sum"),
        total_users     =("user_id",      "count"),
    ).reset_index()

    result = result.merge(totals, on="day")
    col_order = ["day",
                 "match_A", "match_B", "match_C",
                 "different_A", "different_B", "different_C",
                 "dataset_only", "algo_only",
                 "total_with_ref", "total_with_algo", "total_with_home", "total_users"]
    result = result[[c for c in col_order if c in result.columns]]
    return result, df


# ── 3. Cross-comparison table (per pair of methods) ───────────────────────────
def build_cross(df1: pd.DataFrame, df2: pd.DataFrame,
                name1: str, name2: str) -> pd.DataFrame:
    left  = (df1[["user_id", "day", "algo_first", "algo_has"]]
             .rename(columns={"algo_first": "first_1", "algo_has": "has_1"}))
    right = (df2[["user_id", "day", "algo_first", "algo_has"]]
             .rename(columns={"algo_first": "first_2", "algo_has": "has_2"}))
    m = left.merge(right, on=["user_id", "day"])

    def classify(row):
        agree = row["first_1"] == row["first_2"]
        if agree and not row["has_1"]:          return "neither"
        if agree:                               return "agree"
        if row["has_1"] and row["has_2"]:       return "different"
        if row["has_1"]:                        return f"only_{name1}"
        return f"only_{name2}"

    m["case"]         = m.apply(classify, axis=1)
    m["ref_home_has"] = m.apply(lambda r: bool(ref_homes.get((r["user_id"], r["day"]), "")), axis=1)

    case_cols = ["agree", "different", f"only_{name1}", f"only_{name2}"]
    counts = (
        m.groupby(["day", "case"]).size().unstack(fill_value=0)
         .reindex(columns=case_cols, fill_value=0)
         .reset_index()
    )
    totals = m.groupby("day").agg(
        **{f"total_with_{name1}": ("has_1",         "sum"),
           f"total_with_{name2}": ("has_2",         "sum"),
           "total_with_home":     ("ref_home_has",  "sum"),
           "total_users":          ("user_id",      "count")}
    ).reset_index()

    return counts.merge(totals, on="day")


# ── 4. Run everything and collect sheets ──────────────────────────────────────
loaded: dict[str, pd.DataFrame] = {}
sheets: dict[str, pd.DataFrame] = {}   # sheet_name → dataframe

# Short aliases for sheet names (Excel limit: 31 chars)
SHORT = {
    "cont_no_gap":    "cng",
    "cont_gap":       "cg",
    "no_cont_no_gap": "ncng",
    "no_cont_gap":    "ncg",
}

STRAT_LABELS = {
    "A": "A (first in list)",
    "B": "B (most time)",
    "C": "C (most records)",
}

print(f"\n{'═'*70}\n  COMPARISON VS DATASET  (merge={MERGE})\n{'═'*70}")

for method in METHODS:
    df = load_method(method)
    if df is None:
        continue
    loaded[method] = df

    table, df_classified = build_vs_dataset(df)
    sheet_name = f"vs_{SHORT[method]}"
    sheets[sheet_name] = table

    print(f"\n── {method} ──")
    print(table.to_string(index=False))

    # Examples — strategy-independent cases once, then per strategy
    print("  Examples:")
    for case in ["dataset_only", "algo_only"]:
        sub = df_classified[df_classified["case_A"] == case]
        if sub.empty:
            print(f"    {case:15s}: no example")
        else:
            print(f"    {case:15s}:")
            for _, r in sub.head(4).iterrows():
                print(f"      user={r['user_id']}  day={r['day']}"
                      f"  algo={r['algo_cells']}  ref='{r['ref_cell']}'")

    for s, slabel in STRAT_LABELS.items():
        print(f"    Strategy {slabel}:")
        for case in ["match", "different"]:
            sub = df_classified[df_classified[f"case_{s}"] == case]
            if sub.empty:
                print(f"      {case:10s}: no example")
            else:
                print(f"      {case:10s}:")
                for _, r in sub.head(4).iterrows():
                    print(f"        user={r['user_id']}  day={r['day']}"
                          f"  selected={r[f'algo_{s}']}  "
                          f"all_cells={r['algo_cells']}  ref='{r['ref_cell']}'")

print(f"\n{'═'*70}\n  CROSS-METHOD COMPARISONS  (merge={MERGE})\n{'═'*70}")

available = [m for m in METHODS if m in loaded]
for m1, m2 in combinations(available, 2):
    table = build_cross(loaded[m1], loaded[m2], SHORT[m1], SHORT[m2])
    sheet_name = f"cross_{SHORT[m1]}_vs_{SHORT[m2]}"
    sheets[sheet_name] = table

    print(f"\n── {m1}  vs  {m2} ──")
    print(table.to_string(index=False))

# ── 5. Write CSV files ────────────────────────────────────────────────────────
for sheet_name, df in sheets.items():
    path = OUTPUT_DIR / f"{sheet_name}.csv"
    df.to_csv(path, sep=";", index=False)
    print(f"  Saved: {path.name}")

print(f"\nAll files saved in: {OUTPUT_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Categorical palette — one color per method, fixed order
METHOD_COLORS = {
    "cont_no_gap":    "#4C72B0",
    "cont_gap":       "#DD8452",
    "no_cont_no_gap": "#55A868",
    "no_cont_gap":    "#C44E52",
}
METHOD_LABELS = {
    "cont_no_gap":    "Cont. / no gap",
    "cont_gap":       "Cont. / gap",
    "no_cont_no_gap": "No cont. / no gap",
    "no_cont_gap":    "No cont. / gap",
}

# Colors per strategy for multi-strategy plots
STRAT_COLORS = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}

# Colour palette for the 4 case categories (same across all plots)
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

# Collect the vs-dataset tables built earlier
vs_tables = {m: sheets[f"vs_{SHORT[m]}"] for m in METHODS if f"vs_{SHORT[m]}" in sheets}


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Match rate over days per method (strategy A)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
for method, table in vs_tables.items():
    ref = table["total_with_ref"].replace(0, np.nan)
    rate = table["match_A"] / ref
    ax.plot(table["day"], rate, marker="o", markersize=6, linewidth=2,
            color=METHOD_COLORS[method], label=METHOD_LABELS[method])

ax.set_title("Match rate vs dataset per day  — Method A  (no_merge)", fontweight="bold")
ax.set_xlabel("Day")
ax.set_ylabel("Match rate  (match / users with ref cell)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_match_rate_by_day.png", dpi=150)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Stacked breakdown per method (all days combined, strategy A)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(vs_tables))
bottoms = np.zeros(len(vs_tables))
method_list = list(vs_tables.keys())

for case, color in CASE_COLORS.items():
    col = f"{case}_A" if case in ("match", "different") else case
    vals = np.array([vs_tables[m][col].sum() for m in method_list], dtype=float)
    ax.bar(x, vals, bottom=bottoms, color=color, label=CASE_LABELS[case],
           width=0.55, edgecolor="white", linewidth=1.5)
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels([METHOD_LABELS[m] for m in method_list], fontsize=9)
ax.set_title("Overall breakdown per method — Method A  (no_merge)", fontweight="bold")
ax.set_ylabel("Number of user × day pairs")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_breakdown_global_per_method.png", dpi=150)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Activity cell coverage per day
# ─────────────────────────────────────────────────────────────────────────────
ref_table = next(iter(vs_tables.values()))
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(ref_table["day"], ref_table["total_users"],
        color="#333", linewidth=1.5, linestyle=":", label="Total users")
ax.plot(ref_table["day"], ref_table["total_with_ref"],
        color="#888", linewidth=2, linestyle="--", label="Users with ref cell (dataset)")

for method, table in vs_tables.items():
    ax.plot(table["day"], table["total_with_algo"],
            marker="o", markersize=5, linewidth=2,
            color=METHOD_COLORS[method], label=METHOD_LABELS[method])

ax.set_title("Activity cell coverage per day  (no_merge)", fontweight="bold")
ax.set_xlabel("Day")
ax.set_ylabel("Number of users")
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.legend(fontsize=8, ncol=2)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_coverage_by_day.png", dpi=150)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Inter-method agreement heatmap
# ─────────────────────────────────────────────────────────────────────────────
n = len(METHODS)
matrix = np.full((n, n), np.nan)
for i, m1 in enumerate(METHODS):
    matrix[i, i] = 1.0
    for j, m2 in enumerate(METHODS):
        if i >= j:
            continue
        key = f"cross_{SHORT[m1]}_vs_{SHORT[m2]}"
        if key in sheets:
            t = sheets[key]
            # Agreement among the user-days where AT LEAST ONE of the two methods finds an
            # activity cell. Dividing by all users (as before) counted the ~60 % of users
            # for whom neither method finds anything as disagreements.
            both = t["agree"] + t["different"]
            union = t[f"total_with_{SHORT[m1]}"] + t[f"total_with_{SHORT[m2]}"] - both
            rate = t["agree"].sum() / union.sum()
            matrix[i, j] = rate
            matrix[j, i] = rate

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
plt.colorbar(im, ax=ax, fraction=0.046, label="Agreement rate")
labels = [METHOD_LABELS[m] for m in METHODS]
ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
for i in range(n):
    for j in range(n):
        if not np.isnan(matrix[i, j]):
            txt_color = "white" if matrix[i, j] > 0.6 else "#333"
            ax.text(j, i, f"{matrix[i, j]:.1%}", ha="center", va="center",
                    fontsize=10, color=txt_color, fontweight="bold")
ax.set_title("Inter-method agreement rate  (no_merge)\n"
             "among user-days where at least one method finds an activity cell",
             fontweight="bold", fontsize=10)
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_cross_agreement_heatmap.png", dpi=150)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Small multiples: breakdown per day per method (strategy A)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
for ax, method in zip(axes.flat, METHODS):
    if method not in vs_tables:
        ax.set_visible(False)
        continue
    table = vs_tables[method]
    total = table["total_users"].replace(0, np.nan)
    days  = table["day"].tolist()
    x     = np.arange(len(days))

    bottom = np.zeros(len(days))
    for case in ["match", "different", "dataset_only", "algo_only"]:
        col = f"{case}_A" if case in ("match", "different") else case
        vals = (table[col] / total).fillna(0).to_numpy()
        ax.bar(x, vals, bottom=bottom, color=CASE_COLORS[case],
               label=CASE_LABELS[case], width=0.75,
               edgecolor="white", linewidth=0.8)
        bottom += vals

    ax.set_title(METHOD_LABELS[method], fontsize=17, fontweight="bold",
                 color=METHOD_COLORS[method])
    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45, fontsize=7)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

handles, labels_leg = axes[0, 0].get_legend_handles_labels()
# Legend at the bottom, outside the panels, so it never overlaps the title
fig.legend(handles, labels_leg, loc="lower center", ncol=4, fontsize=13, frameon=False)
fig.suptitle(
    "Case breakdown per day per method — Method A  (% of total users, no_merge)",
    fontsize=15, fontweight="bold"
)
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig(PLOT_DIR / "05_breakdown_per_day_per_method.png", dpi=150)
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Strategy comparison (A vs B vs C) : match rate per day per method
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
for ax, method in zip(axes.flat, METHODS):
    if method not in vs_tables:
        ax.set_visible(False)
        continue
    table = vs_tables[method]
    ref   = table["total_with_ref"].replace(0, np.nan)
    for s, slabel in STRAT_LABELS.items():
        rate = table[f"match_{s}"] / ref
        ax.plot(table["day"], rate, marker="o", markersize=5, linewidth=2,
                color=STRAT_COLORS[s], label=f"Method {slabel}")

    ax.set_title(METHOD_LABELS[method], fontsize=14, fontweight="bold",
                 color=METHOD_COLORS[method])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

handles, labels_leg = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc="upper left", ncol=1, fontsize=12, frameon=False)
fig.suptitle(
    "Match rate per day — A (first) vs B (most time) vs C (most records)  (no_merge)",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(PLOT_DIR / "06_strategy_comparison_match_rate.png", dpi=150)
plt.close()

print(f"Plots saved in: {PLOT_DIR}")
