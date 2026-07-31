from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from maximal_previsibility import load_pmax_dataframe

"""
Day-by-day summary of the sequential (causal) next-cell prediction: for each
day, the mean accuracy actually reached by the online Markov model next to the
mean theoretical ceiling P^max over the same users.

This replaces the per-user example figures, which were unreadable in practice
(users visit too many distinct cells for a single-day plot to be legible).
"""

MAIN_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = MAIN_DIR / "results/numpy"
OUTPUT_DIR = MAIN_DIR / "results/plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DAY_ABBR = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def is_weekend(day_str):
    return datetime.strptime(day_str, "%Y-%m-%d").weekday() >= 5


def load_results():
    results_path = RESULTS_DIR / "markov_sequential_accuracy_no_merge.npy"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found - run markov_sequential_prediction.py first."
        )

    raw = np.load(results_path, allow_pickle=True).item()
    rows = [{'id_user': id_user, 'day': values[0], 'accuracy': values[1],
             'n_transitions': values[2], 'n_records': values[3]}
            for id_user, values in raw.items()]
    df = pd.DataFrame(rows)

    # Inner join: load_pmax_dataframe() already drops zero-entropy users, so
    # they are excluded from the day-by-day means too.
    pmax_df = load_pmax_dataframe()[['id_user', 'pmax']]
    df = df.merge(pmax_df, on='id_user', how='inner')
    df['gap'] = df['pmax'] - df['accuracy']
    return df


def summarise_by_day(df):
    by_day = df.groupby('day').agg(
        n_users=('id_user', 'count'),
        accuracy_mean=('accuracy', 'mean'),
        accuracy_median=('accuracy', 'median'),
        pmax_mean=('pmax', 'mean'),
        gap_mean=('gap', 'mean'),
    ).sort_index()
    return by_day


def plot_by_day(by_day):
    days = by_day.index.tolist()
    x = np.arange(len(days))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    bars_acc = ax.bar(x - width / 2, by_day['accuracy_mean'], width,
                      label="Sequential Markov accuracy", color="seagreen")
    bars_pmax = ax.bar(x + width / 2, by_day['pmax_mean'], width,
                       label="$P^{max}$ (theoretical ceiling)", color="orange")

    for bars in (bars_acc, bars_pmax):
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45, ha="right")
    for tick, day in zip(ax.get_xticklabels(), days):
        if is_weekend(day):
            tick.set_color("red")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Day (weekends in red)")
    ax.set_ylabel("Mean over the users of that day")
    ax.set_title("Sequential next-cell prediction vs. theoretical ceiling, day by day")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_calendar(df, x_col, y_col, x_short, y_short, x_label, y_label,
                  suptitle, filename):
    """One panel per day, laid out as a real calendar (columns = weekday), each
    panel a hexbin of y_col against x_col.

    A single scatter over the ~740k users is a saturated blob. Splitting by day
    and colouring by bin density instead of plotting one dot per user makes the
    structure readable, and the calendar layout lines the weekends up in the two
    right-hand columns so the weekend effect is visible at a glance.

    Generic in the two columns plotted so it can serve both the accuracy vs
    P^max view and the sequential vs random-split comparison. x_short/y_short
    are the compact names used in the per-panel titles.
    """
    days = sorted(df['day'].unique())
    dates = {day: datetime.strptime(day, "%Y-%m-%d") for day in days}

    # Row = week number counted from the Monday of the first week, column = weekday.
    first_monday = dates[days[0]] - timedelta(days=dates[days[0]].weekday())
    position = {day: ((dates[day] - first_monday).days // 7, dates[day].weekday())
                for day in days}
    n_rows = max(row for row, _ in position.values()) + 1

    fig, axes = plt.subplots(n_rows, 7, figsize=(17, 3.1 * n_rows),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_visible(False)

    hexbin = None
    for day in days:
        row, col = position[day]
        ax = axes[row, col]
        ax.set_visible(True)

        sub = df[df['day'] == day]
        hexbin = ax.hexbin(sub[x_col], sub[y_col], gridsize=26, cmap="Blues",
                           bins="log", extent=(0, 1, 0, 1), mincnt=1)
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)

        # Kept short and on three lines: a single-line title overflows into the
        # neighbouring panel at this grid width.
        weekday = dates[day].weekday()
        ax.set_title(f"{day} ({DAY_ABBR[weekday]})\nn={len(sub)}\n"
                     f"{y_short} {sub[y_col].mean():.2f} - {x_short} {sub[x_col].mean():.2f}",
                     color="red" if weekday >= 5 else "black",
                     fontsize=8, fontweight="bold", pad=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.colorbar(hexbin, ax=axes.ravel().tolist(), label="Users per bin (log scale)",
                 shrink=0.6)
    fig.supxlabel(x_label)
    fig.supylabel(y_label)
    fig.suptitle(suptitle, fontsize=14)

    fig.savefig(OUTPUT_DIR / filename, dpi=110)
    plt.show()


if __name__ == "__main__":
    df = load_results()
    by_day = summarise_by_day(df)

    print(by_day.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nOverall mean accuracy : {df['accuracy'].mean():.3f}")
    print(f"Overall mean P^max    : {df['pmax'].mean():.3f}")
    print(f"Overall mean gap      : {df['gap'].mean():.3f}")

    plot_by_day(by_day)
    plot_calendar(df, 'pmax', 'accuracy', 'Pmax', 'acc',
                  "Theoretical $P^{max}$ (Fano)", "Sequential Markov accuracy",
                  "Sequential next-cell prediction vs. $P^{max}$ - one panel per day "
                  "(dashed line: y = x)",
                  "markov_sequential_vs_pmax_calendar.png")
