from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from maximal_previsibility import load_user_entropies
from plot_markov_sequential_by_day import is_weekend, plot_calendar

"""
Side-by-side comparison of the two prediction protocols, without any reference
to P^max:

  - sequential (markov_sequential_prediction.py): at each step the model only
    knows the transitions that happened strictly before, and learns online.
    This is the real objective - predicting the next station from the past only.

  - random split (markov_baseline.py): 70% of the user's transitions drawn at
    random are used to build the model, the remaining 30% to test it. The model
    therefore sees transitions from later in the day, which the sequential one
    cannot.

Only users present in both runs are kept, so the two numbers are always
computed over exactly the same population. Zero-entropy users (never left a
single cell) are excluded, as everywhere else in this comparison.
"""

MAIN_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = MAIN_DIR / "results/numpy"
OUTPUT_DIR = MAIN_DIR / "results/plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_one(filename, accuracy_column):
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the matching script first.")
    raw = np.load(path, allow_pickle=True).item()
    return pd.DataFrame([{'id_user': id_user, 'day': values[0],
                          accuracy_column: values[1]}
                         for id_user, values in raw.items()])


def load_results():
    seq = load_one("markov_sequential_accuracy_no_merge.npy", "accuracy_seq")
    rnd = load_one("markov_accuracy_no_merge.npy", "accuracy_rnd")

    df = seq.merge(rnd.drop(columns='day'), on='id_user', how='inner')

    kept = set(load_user_entropies()['id_user'])
    df = df[df['id_user'].isin(kept)].reset_index(drop=True)

    df['difference'] = df['accuracy_rnd'] - df['accuracy_seq']
    return df


def summarise_by_day(df):
    return df.groupby('day').agg(
        n_users=('id_user', 'count'),
        seq_mean=('accuracy_seq', 'mean'),
        seq_median=('accuracy_seq', 'median'),
        rnd_mean=('accuracy_rnd', 'mean'),
        rnd_median=('accuracy_rnd', 'median'),
        diff_mean=('difference', 'mean'),
    ).sort_index()


def plot_by_day(by_day):
    days = by_day.index.tolist()
    x = np.arange(len(days))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    bars_seq = ax.bar(x - width / 2, by_day['seq_mean'], width,
                      label="Sequential (past only)", color="seagreen")
    bars_rnd = ax.bar(x + width / 2, by_day['rnd_mean'], width,
                      label="Random 70/30 split", color="steelblue")

    for bars in (bars_seq, bars_rnd):
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45, ha="right")
    for tick, day in zip(ax.get_xticklabels(), days):
        if is_weekend(day):
            tick.set_color("red")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Day (weekends in red)")
    ax.set_ylabel("Mean accuracy over the users of that day")
    ax.set_title("Next-cell prediction: sequential vs. random-split, day by day")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "markov_methods_by_day.png", dpi=110)
    plt.show()


def plot_difference(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['difference'], bins=120, color="darkorange", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(df['difference'].mean(), color="red", linestyle="--",
               label=f"Mean: {df['difference'].mean():+.3f}")
    ax.set_xlabel("Random-split accuracy - sequential accuracy (per user)")
    ax.set_ylabel("Number of users")
    ax.set_title("How much the random split gains by also seeing the rest of the day")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "markov_methods_difference.png", dpi=110)
    plt.show()


if __name__ == "__main__":
    df = load_results()
    by_day = summarise_by_day(df)

    print(by_day.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nUsers compared             : {len(df)}")
    print(f"Mean accuracy sequential   : {df['accuracy_seq'].mean():.3f}"
          f"   (median {df['accuracy_seq'].median():.3f})")
    print(f"Mean accuracy random split : {df['accuracy_rnd'].mean():.3f}"
          f"   (median {df['accuracy_rnd'].median():.3f})")
    print(f"Mean difference            : {df['difference'].mean():+.3f}")
    print(f"Users where sequential wins: {(df['accuracy_seq'] > df['accuracy_rnd']).mean():.1%}")
    print(f"Correlation between methods: {df['accuracy_seq'].corr(df['accuracy_rnd']):.3f}")

    plot_by_day(by_day)
    plot_difference(df)
    plot_calendar(df, 'accuracy_rnd', 'accuracy_seq', 'rnd', 'seq',
                  "Random-split accuracy", "Sequential accuracy",
                  "Sequential vs. random-split accuracy - one panel per day "
                  "(dashed line: y = x)",
                  "markov_methods_calendar.png")
