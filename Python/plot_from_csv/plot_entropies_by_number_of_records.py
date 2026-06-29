import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict
import datetime

MAIN_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = MAIN_DIR / "results/numpy"
OUTPUT_DIR = MAIN_DIR / "results/plots"

WEEKEND_DATES = {"2014-03-15", "2014-03-16", "2014-03-22", "2014-03-23"}

# One color per day of week (Monday=0 ... Sunday=6)
DAY_COLORS = {
    0: '#1f77b4',  # Monday
    1: '#ff7f0e',  # Tuesday
    2: '#2ca02c',  # Wednesday
    3: '#9467bd',  # Thursday
    4: '#8c564b',  # Friday
    5: '#e377c2',  # Saturday
    6: '#bcbd22',  # Sunday
}
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# 1. Load numpy file
data = np.load(INPUT_DIR / 'user_entropies_no_merge.npy', allow_pickle=True).item()

# 2. Group data by day
day_data = defaultdict(lambda: {'records': [], 'entropy': [], 'relative_entropy': []})

for id_user, values in data.items():
    day, user_entropy, user_relative_entropy, num_record = values
    day_str = str(day)
    if user_entropy != 0:
        day_data[day_str]['records'].append(num_record)
        day_data[day_str]['entropy'].append(user_entropy)
        day_data[day_str]['relative_entropy'].append(user_relative_entropy)

days = sorted(day_data.keys())
n_days = len(days)
n_cols = 5
n_rows = (n_days + n_cols - 1) // n_cols

# Legend handles (one per day of week present in data)
weekdays_present = sorted({datetime.datetime.strptime(d, "%Y-%m-%d").weekday() for d in days})
legend_handles = [
    mpatches.Patch(color=DAY_COLORS[wd], label=DAY_NAMES[wd])
    for wd in weekdays_present
]


def make_figure(metric_key, ylabel, suptitle):
    fig, axes = plt.subplots(n_rows, n_cols, sharey=True,constrained_layout=True,)
    axes = axes.flatten() if n_days > 1 else [axes]
    fig.suptitle(suptitle, fontsize=14)

    for i, day in enumerate(days):
        ax = axes[i]
        d = day_data[day]
        date_obj = datetime.datetime.strptime(day, "%Y-%m-%d")
        weekday = date_obj.weekday()

        ax.scatter(d['records'], d[metric_key],
                   color=DAY_COLORS[weekday], alpha=0.4, edgecolors='none', s=20)

        title_color = 'red' if day in WEEKEND_DATES else 'black'
        ax.set_title(f"{day} ({DAY_NAMES[weekday]})", color=title_color, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Number of Records (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.grid(True, linestyle='--', alpha=0.6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    return fig


# 3. Figure 1 — Entropy per day
make_figure('entropy', 'Entropy', 'User Entropy vs. Number of Records — per day')
plt.savefig(OUTPUT_DIR / "entropy_by_user_by_day.png")

# 4. Figure 2 — Relative entropy per day
make_figure('relative_entropy', 'Relative Entropy',
            'User Relative Entropy vs. Number of Records — per day')
plt.savefig(OUTPUT_DIR / "relative_entropy_by_user_by_day.png")



plt.show()