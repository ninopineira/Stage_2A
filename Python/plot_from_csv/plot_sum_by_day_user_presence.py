import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from pathlib import Path
import csv
from datetime import datetime

MAIN_DIR = Path(__file__).parent.parent.parent
STATS_DIR = MAIN_DIR / "Database/no_duplicate"
OUTPUT_DIR = MAIN_DIR / "results/plots"

def get_day(filepath : Path) -> str:
    return filepath.name.split("_")[0]

files = [file for file in STATS_DIR.glob("*.csv")]

def get_user_occupancy(user_stamps: list[int], ocupancy_by_hour: dict[int, int]) -> dict[int, int]:
    passed_hours = {i: False for i in range(24)}
    for stamp in user_stamps:
        hour = stamp // 3600
        if not passed_hours[hour]:
            ocupancy_by_hour[hour] += 1
            passed_hours[hour] = True
    return ocupancy_by_hour

N_COLS = 5
n_days = 15
n_rows = (n_days + N_COLS - 1) // N_COLS

fig, axes = plt.subplots(
n_rows, N_COLS,
figsize=(N_COLS * 3.5, n_rows * 3),
sharey=True,
constrained_layout=True,
)
axes = axes.flatten()

# Mon, Tue, Wed, Thu, Fri, Sat, Sun
DAY_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#E377C2", "#7F7F7F"]

for i, file in enumerate(files):
    day = get_day(file)
    weekday = datetime.strptime(day, "%Y-%m-%d").weekday()
    ocupancy_by_hour = {hour: 0 for hour in range(24)}
    with open(file, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            user_id = line[0]
            user_cells = [c for c in line[8::2] if c]
            user_stamps = [int(ts) for ts in line[9::2] if ts]

            ocupancy_by_hour = get_user_occupancy(user_stamps, ocupancy_by_hour)

    ax = axes[i]
    ax.bar(ocupancy_by_hour.keys(), ocupancy_by_hour.values(), color=DAY_COLORS[weekday], width=0.7)
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
        ax.set_ylabel(f"Sum of Users", fontsize=8)

plt.suptitle("Sum of users present in the area by hour and by day", fontsize=12, fontweight="bold")
plt.savefig(OUTPUT_DIR / "sum_by_day_user_presence.png")
plt.show() 

    


