import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from pathlib import Path

MAIN_DIR = Path(__file__).parent.parent.parent
STATS_DIR = MAIN_DIR / "results/intermediate_result"
OUTPUT_DIR = MAIN_DIR / "results/plots"

"""
The aim of this script is to plot the sum of users and connections by day and by hour, for each of the 3 merge methods (no merge, simple merge, 2g/3g merge).
The output are 6 plots of 15 graphes (one for each merge method and each metric) showing the sum of users and connections by day and by hour.
"""

files = [
    file for file in STATS_DIR.glob("*.csv")
    if "stats_connections_cell_by_hour" in file.name]

N_COLS = 5

for file in files[1:2]:
    df = pd.read_csv(file, sep=";")
    df_grouped = df.groupby("day").sum()

    days = df_grouped.index.tolist()
    hours = df_grouped.columns[1:]
    n_days = len(days)
    n_rows = (n_days + N_COLS - 1) // N_COLS

    is_users = "use_cell" in file.name
    metric_label = "Users" if is_users else "Connections"

    colors = cm.tab20.colors

    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(N_COLS * 3.5, n_rows * 3),
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.flatten()

    for i, day in enumerate(days):
        ax = axes[i]
        ax.bar(hours, df_grouped.loc[day].iloc[1:], color=colors[i % len(colors)], width=0.7)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_xlabel("Hour", fontsize=8)
        weekend = ["2014-03-15", "2014-03-16", "2014-03-22", "2014-03-23"]
        if day in weekend:
            ax.set_title(day, color="#ff0000", fontsize=9, fontweight="bold", pad=4)
        else:
            ax.set_title(day, fontsize=9, fontweight="bold", pad=4)
    
    if i % N_COLS == 0:
        if i % N_COLS == 0:
            ax.set_ylabel(f"Sum of {metric_label}", fontsize=8)

    for j in range(n_days, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Sum of {metric_label} by hour and day — {file.stem}",
        fontsize=13, fontweight="bold",
    )

    plt.savefig(OUTPUT_DIR / f"{file.stem}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
