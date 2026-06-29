import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

MAIN_DIR   = Path(__file__).parent.parent.parent
NUMPY_DIR  = MAIN_DIR / "results/numpy"
OUTPUT_DIR = MAIN_DIR / "results/plots"

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
MERGE        = "no_merge"   # "no_merge" | "simple" | "2g3g"
ENTROPY_TYPE = 0            # 0 = classique | 1 = normalisee

PERIOD_NAMES   = ["Matin", "Jour", "Soir"]
MORNING_ENDS   = [4, 5, 6]
EVENING_STARTS = [18, 19, 20]
TYPE_NAMES     = ["classique", "normalisee"]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD  {day: [[55 values per user], ...]}
#   values 0-26  : classique entropies
#   values 27-53 : normalisee entropies
#   value  54    : total nb. of records for that user/day
# ─────────────────────────────────────────────────────────────────────────────
raw = np.load(NUMPY_DIR / f"user_entropies_by_period_{MERGE}.npy",
              allow_pickle=True).item()

DAY_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#E377C2", "#7F7F7F"]
WEEKENDS   = {5, 6}

y_label = "Entropie normalisee" if ENTROPY_TYPE == 1 else "Entropie (bits)"

# ─────────────────────────────────────────────────────────────────────────────
# 3 PLOTS PER DAY  (one per period: matin / jour / soir)
# each plot = 3x3 grid of morning_end x evening_start combinations
# ─────────────────────────────────────────────────────────────────────────────
for day in sorted(raw.keys()):
    users = raw.get(day, [])
    if not users:
        print(f"  {day}: no data, skipped")
        continue

    arr = np.array(users, dtype=float)          # (n_users, 55)
    x_all = arr[:, 54]                          # num_records (all users)

    dt      = datetime.strptime(day, "%Y-%m-%d")
    weekday = dt.weekday()
    color   = DAY_COLORS[weekday]
    is_we   = weekday in WEEKENDS
    title_color = "#c0392b" if is_we else "#222"

    day_dir = OUTPUT_DIR / day
    day_dir.mkdir(parents=True, exist_ok=True)

    for period in range(3):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
        fig.suptitle(
            f"{day}  —  {PERIOD_NAMES[period]} | merge: {MERGE} | {TYPE_NAMES[ENTROPY_TYPE]}",
            fontsize=13, fontweight="bold", color=title_color
        )

        for mi in range(3):                     # morning_end index
            for ei in range(3):                 # evening_start index
                combo_idx = mi * 3 + ei
                ent_idx   = combo_idx * 3 + period + (27 if ENTROPY_TYPE == 1 else 0)

                ax    = axes[mi][ei]
                valid = arr[:, ent_idx] > 0
                x     = x_all[valid]
                y     = arr[valid, ent_idx]

                ax.scatter(x, y, s=3, alpha=0.35, color=color, linewidths=0)
                ax.set_xscale("log")
                ax.set_xlabel("Nb. records (log)", fontsize=8)
                ax.set_ylabel(y_label, fontsize=8)
                ax.set_title(
                    f"Matin→{MORNING_ENDS[mi]}h  |  Soir←{EVENING_STARTS[ei]}h",
                    fontsize=9
                )
                ax.tick_params(labelsize=7)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="both", linestyle="--", alpha=0.3)

        fname = f"entropy_scatter_{MERGE}_{PERIOD_NAMES[period].lower()}_{TYPE_NAMES[ENTROPY_TYPE]}.png"
        plt.savefig(day_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {day}/{fname}")

print("Done.")
