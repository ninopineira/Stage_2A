import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

MAIN_DIR   = Path(__file__).parent.parent.parent
NUMPY_DIR  = MAIN_DIR / "results/numpy"
OUTPUT_DIR = MAIN_DIR / "results/plots"

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
MERGE        = "no_merge"   # "no_merge" | "simple" | "2g3g"
ENTROPY_TYPE = 1            # 0 = classic | 1 = normalised

PERIOD_NAMES   = ["Morning", "Day", "Evening"]
PERIOD_COLORS  = ["#4C72B0", "#DD8452", "#55A868"]   # blue, orange, green
MORNING_ENDS   = [4, 5, 6]
EVENING_STARTS = [18, 19, 20]
ENTROPY_LABELS = ["Uncorrelated_temporal_entropy", "Relative_entropy"]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD  {day: [[55 values per user], ...]}
#   values 0-26  : classic entropies    (9 combos × 3 sub-periods)
#   values 27-53 : normalised entropies
#   value  54    : total number of records for that user/day
# ─────────────────────────────────────────────────────────────────────────────
raw = np.load(NUMPY_DIR / f"user_entropies_by_period_{MERGE}.npy",
              allow_pickle=True).item()

DAY_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#E377C2", "#7F7F7F"]
WEEKENDS   = {5, 6}

y_label      = ENTROPY_LABELS[ENTROPY_TYPE]
entropy_name = ENTROPY_LABELS[ENTROPY_TYPE]

# ─────────────────────────────────────────────────────────────────────────────
# ONE PLOT PER DAY
#   3×3 grid  →  rows = morning end (4h, 5h, 6h)
#                cols = evening start (18h, 19h, 20h)
#   each subplot  →  3 scatter clouds: Morning / Day / Evening
# ─────────────────────────────────────────────────────────────────────────────
for day in sorted(raw.keys()):
    users = raw.get(day, [])
    if not users:
        print(f"  {day}: no data, skipped")
        continue

    arr   = np.array(users, dtype=float)    # (n_users, 55)
    x_all = arr[:, 54]                      # number of records

    dt          = datetime.strptime(day, "%Y-%m-%d")
    weekday     = dt.weekday()
    is_weekend  = weekday in WEEKENDS
    title_color = "#c0392b" if is_weekend else "#222"

    day_dir = OUTPUT_DIR / day
    day_dir.mkdir(parents=True, exist_ok=True)

    for mi in range(3):                             # morning end index
        for ei in range(3):                         # evening start index
            combo_idx = mi * 3 + ei

            fig = plt.figure(figsize=(19, 4), constrained_layout=True)
            gs  = fig.add_gridspec(1, 4, width_ratios=[1, 3, 3, 3])

            ax_info  = fig.add_subplot(gs[0, 0])
            ref_ax   = fig.add_subplot(gs[0, 1])
            axes     = [ref_ax,
                        fig.add_subplot(gs[0, 2], sharex=ref_ax, sharey=ref_ax),
                        fig.add_subplot(gs[0, 3], sharex=ref_ax, sharey=ref_ax)]

            ax_info.set_axis_off()
            ax_info.text(
                0.5, 0.5,
                f"Morning end:    {MORNING_ENDS[mi]}h\n"
                f"Evening start:  {EVENING_STARTS[ei]}h\n"
                f"Merge:  {MERGE}\n\n"
                f"{entropy_name}",
                transform=ax_info.transAxes, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.7", facecolor="#f5f5f5",
                          edgecolor="#bbbbbb", alpha=0.9)
            )
            fig.suptitle(day, fontsize=21, fontweight="bold", color=title_color)

            for period in range(3):
                ent_idx = combo_idx * 3 + period + (27 if ENTROPY_TYPE == 1 else 0)
                valid   = arr[:, ent_idx] > 0
                x       = x_all[valid]
                y       = arr[valid, ent_idx]

                ax = axes[period]
                ax.scatter(x, y, s=3, alpha=0.35,
                           color=PERIOD_COLORS[period], linewidths=0,
                           label=entropy_name)
                ax.set_xscale("log")
                ax.set_xlabel("Nb. records (log scale)", fontsize=9)
                ax.set_ylabel(y_label, fontsize=9)
                ax.set_title(PERIOD_NAMES[period], fontsize=16,
                             color=PERIOD_COLORS[period], fontweight="bold")
                ax.tick_params(labelsize=8)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="both", linestyle="--", alpha=0.3)

            fname = (f"entropy_scatter_{MERGE}_{ENTROPY_LABELS[ENTROPY_TYPE]}"
                     f"_morning{MORNING_ENDS[mi]}h_evening{EVENING_STARTS[ei]}h.png")
            plt.savefig(day_dir / fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {day}/{fname}")

print("Done.")

# ─────────────────────────────────────────────────────────────────────────────
# CALENDAR DISPLAY  —  fixed window: morning end 6h, evening start 19h
#   3 figures (one per period: Morning / Day / Evening)
#   each figure: calendar grid (n_weeks × 7), one scatter per day cell
# ─────────────────────────────────────────────────────────────────────────────
CAL_MI, CAL_EI = 2, 1                    # morning end 6h, evening start 19h
CAL_COMBO      = CAL_MI * 3 + CAL_EI    # = 7

days_sorted = sorted(raw.keys())
dt_days     = [datetime.strptime(d, "%Y-%m-%d") for d in days_sorted]
first_monday = dt_days[0] - timedelta(days=dt_days[0].weekday())
n_weeks      = ((dt_days[-1] - first_monday).days // 7) + 1

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

for period in range(3):
    ent_idx = CAL_COMBO * 3 + period + (27 if ENTROPY_TYPE == 1 else 0)

    fig = plt.figure(figsize=(26, n_weeks * 3.5), constrained_layout=True)
    gs  = fig.add_gridspec(n_weeks, 8, width_ratios=[1.5, 3, 3, 3, 3, 3, 3, 3])

    ax_info = fig.add_subplot(gs[:, 0])
    ax_info.set_axis_off()
    ax_info.text(
        0.5, 0.5,
        f"Morning end:    {MORNING_ENDS[CAL_MI]}h\n"
        f"Evening start:  {EVENING_STARTS[CAL_EI]}h\n"
        f"Merge:  {MERGE}\n\n"
        f"{entropy_name}",
        transform=ax_info.transAxes, ha="center", va="center", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f5f5f5",
                  edgecolor="#bbbbbb", alpha=0.9)
    )
    fig.suptitle(PERIOD_NAMES[period], fontsize=21, fontweight="bold",
                 color=PERIOD_COLORS[period])

    ref_ax = None
    axes   = []
    for r in range(n_weeks):
        row = []
        for c in range(7):
            kw = {} if ref_ax is None else {"sharey": ref_ax}
            ax = fig.add_subplot(gs[r, c + 1], **kw)
            if ref_ax is None:
                ref_ax = ax
            row.append(ax)
        axes.append(row)

    # hide all cells by default
    for r in range(n_weeks):
        for c in range(7):
            axes[r][c].set_visible(False)

    for day, dt in zip(days_sorted, dt_days):
        row = (dt - first_monday).days // 7
        col = dt.weekday()
        is_we = col in WEEKENDS

        ax = axes[row][col]
        ax.set_visible(True)

        users = raw.get(day, [])
        if users:
            arr_day = np.array(users, dtype=float)
            valid   = arr_day[:, ent_idx] > 0
            x = arr_day[valid, 54]
            y = arr_day[valid, ent_idx]
            ax.scatter(x, y, s=2, alpha=0.3,
                       color=DAY_COLORS[col], linewidths=0)
            ax.set_xscale("log")

        ax.set_title(f"{WEEKDAY_NAMES[col]} {day}", fontsize=13, fontweight="bold",
                     color="#c0392b" if is_we else "#222", pad=2)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="both", linestyle="--", alpha=0.3)
        ax.set_xlabel("Records", fontsize=10)

    # Enable Y ticks/label on the leftmost visible cell of each row
    for r in range(n_weeks):
        for c in range(7):
            ax = axes[r][c]
            if ax.get_visible():
                ax.tick_params(labelleft=True)
                ax.set_ylabel(y_label, fontsize=10)
                break

    fname = (f"entropy_calendar_{MERGE}_{ENTROPY_LABELS[ENTROPY_TYPE]}"
             f"_morning{MORNING_ENDS[CAL_MI]}h_evening{EVENING_STARTS[CAL_EI]}h"
             f"_{PERIOD_NAMES[period].lower()}.png")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved calendar: {fname}")

print("Calendar done.")

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS PER DAY OF THE WEEK
#   Same window as the calendar (CAL_MI, CAL_EI), all 3 periods.
#   Metrics: n, mean, std, median, Q1, Q3, IQR, min, max
# ─────────────────────────────────────────────────────────────────────────────
from collections import defaultdict

# Accumulate entropy values: weekday_data[period][weekday] = list of values
weekday_data = [[defaultdict(list) for _ in range(7)] for _ in range(3)]

for day, dt in zip(days_sorted, dt_days):
    users = raw.get(day, [])
    if not users:
        continue
    arr_day = np.array(users, dtype=float)
    wd = dt.weekday()
    for period in range(3):
        ent_idx = CAL_COMBO * 3 + period + (27 if ENTROPY_TYPE == 1 else 0)
        valid   = arr_day[:, ent_idx] > 0
        weekday_data[period][wd]["values"].extend(arr_day[valid, ent_idx].tolist())

STATS_HEADER = f"\n{'=' * 80}\nSTATISTICS PER DAY OF THE WEEK — {entropy_name} | merge={MERGE}\n{'=' * 80}"
print(STATS_HEADER)

COL_W = 10
METRICS = ["n", "mean", "std", "median", "Q1", "Q3", "IQR", "min", "max"]

for period in range(3):
    print(f"\n  ── {PERIOD_NAMES[period]} ──")
    header = f"  {'Metric':<10}" + "".join(f"{WEEKDAY_NAMES[wd]:>{COL_W}}" for wd in range(7))
    print(header)
    print("  " + "-" * (10 + 7 * COL_W))

    rows = {m: [] for m in METRICS}
    for wd in range(7):
        vals = np.array(weekday_data[period][wd].get("values", []))
        if len(vals) == 0:
            for m in METRICS:
                rows[m].append("—")
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        rows["n"].append(str(len(vals)))
        rows["mean"].append(f"{vals.mean():.4f}")
        rows["std"].append(f"{vals.std():.4f}")
        rows["median"].append(f"{med:.4f}")
        rows["Q1"].append(f"{q1:.4f}")
        rows["Q3"].append(f"{q3:.4f}")
        rows["IQR"].append(f"{q3 - q1:.4f}")
        rows["min"].append(f"{vals.min():.4f}")
        rows["max"].append(f"{vals.max():.4f}")

    for m in METRICS:
        print(f"  {m:<10}" + "".join(f"{v:>{COL_W}}" for v in rows[m]))

print(f"\n{'=' * 80}\n")
