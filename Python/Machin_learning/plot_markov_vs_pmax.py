from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from maximal_previsibility import load_pmax_dataframe

"""
Compares the empirical accuracy of the personal Markov baseline
(markov_baseline.py) to the theoretical predictability ceiling P^max
(maximal_previsibility.py) for the same users.
"""

MAIN_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = MAIN_DIR / "results/numpy"

pmax_df = load_pmax_dataframe()

raw_accuracy = np.load(INPUT_DIR / "markov_accuracy_no_merge.npy", allow_pickle=True).item()
accuracy_rows = []
for id_user, values in raw_accuracy.items():
    day, accuracy, n_train, n_test, n_records = values
    accuracy_rows.append({'id_user': id_user, 'accuracy': accuracy,
                          'n_train': n_train, 'n_test': n_test})
accuracy_df = pd.DataFrame(accuracy_rows)

df = pmax_df.merge(accuracy_df, on='id_user', how='inner').dropna(subset=['pmax'])

# ── Statistics ──────────────────────────────────────────────────────────────
print(f"Users compared        : {len(df)}")
print(f"Mean accuracy         : {df['accuracy'].mean():.3f}")
print(f"Mean Pmax             : {df['pmax'].mean():.3f}")
print(f"Median accuracy       : {df['accuracy'].median():.3f}")
print(f"Median Pmax           : {df['pmax'].median():.3f}")
print(f"Correlation acc/pmax  : {df['accuracy'].corr(df['pmax']):.3f}")
print(f"% users acc > pmax    : {(df['accuracy'] > df['pmax']).mean() * 100:.1f}% (sanity check, should be ~0%)")

# ── Scatter: accuracy vs Pmax ────────────────────────────────────────────────
plt.figure(figsize=(7, 7))
plt.scatter(df['pmax'], df['accuracy'], s=5, alpha=0.2, color='steelblue')
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label="y = x")
plt.xlabel("Theoretical $P^{max}$ (Fano)")
plt.ylabel("Empirical Markov accuracy")
plt.title("Personal Markov accuracy vs. theoretical predictability ceiling")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# ── Overlaid histograms ───────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.hist(df['pmax'], bins=100, alpha=0.5, label=f"$P^{{max}}$ (mean {df['pmax'].mean():.2f})", color='orange')
plt.hist(df['accuracy'], bins=100, alpha=0.5, label=f"Markov accuracy (mean {df['accuracy'].mean():.2f})", color='steelblue')
plt.axvline(0.93, color='red', linestyle='--', label="Song et al.: 0.93")
plt.xlabel("Predictability")
plt.ylabel("Number of Users")
plt.title("Distribution of $P^{max}$ vs. empirical Markov accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# ── Accuracy / Pmax by number of records ─────────────────────────────────────
RECORD_BINS = [10, 20, 50, 100, 200, np.inf]
RECORD_LABELS = ["10-20", "20-50", "50-100", "100-200", "200+"]
df['n_records_bucket'] = pd.cut(df['n_records'], bins=RECORD_BINS,
                                 labels=RECORD_LABELS, right=False)

by_bucket = df.groupby('n_records_bucket', observed=True).agg(
    accuracy_mean=('accuracy', 'mean'),
    pmax_mean=('pmax', 'mean'),
    n_users=('id_user', 'count'),
).reindex(RECORD_LABELS)

x = np.arange(len(by_bucket))
width = 0.35
plt.figure(figsize=(10, 5))
plt.bar(x - width / 2, by_bucket['accuracy_mean'], width, label='Markov accuracy', color='steelblue')
plt.bar(x + width / 2, by_bucket['pmax_mean'], width, label='$P^{max}$', color='orange')
for i, n in enumerate(by_bucket['n_users']):
    plt.text(i, max(by_bucket['accuracy_mean'].iloc[i], by_bucket['pmax_mean'].iloc[i]) + 0.02,
             f"n={n}", ha='center', fontsize=8)
plt.xticks(x, RECORD_LABELS)
plt.xlabel("Number of records (day)")
plt.ylabel("Mean value")
plt.title("Accuracy and $P^{max}$ by number of records per user")
plt.legend()
plt.tight_layout()
plt.show()

# ── Accuracy / Pmax by day (weekends highlighted) ───────────────────────────
def is_weekend(day_str):
    return datetime.strptime(day_str, "%Y-%m-%d").weekday() >= 5

by_day = df.groupby('day').agg(
    accuracy_mean=('accuracy', 'mean'),
    pmax_mean=('pmax', 'mean'),
).sort_index()

days = by_day.index.tolist()
x = np.arange(len(days))
width = 0.35
plt.figure(figsize=(12, 5))
plt.bar(x - width / 2, by_day['accuracy_mean'], width, label='Markov accuracy', color='steelblue')
plt.bar(x + width / 2, by_day['pmax_mean'], width, label='$P^{max}$', color='orange')
plt.xticks(x, days, rotation=45, ha='right')
for tick, day in zip(plt.gca().get_xticklabels(), days):
    if is_weekend(day):
        tick.set_color('red')
plt.xlabel("Day (weekends in red)")
plt.ylabel("Mean value")
plt.title("Accuracy and $P^{max}$ by day")
plt.legend()
plt.tight_layout()
plt.show()

# ── Gap between theoretical ceiling and empirical accuracy ─────────────────
df['gap'] = df['pmax'] - df['accuracy']
plt.figure(figsize=(10, 5))
plt.hist(df['gap'], bins=100, color='seagreen', edgecolor='white')
plt.axvline(df['gap'].mean(), color='red', linestyle='--',
            label=f"Mean gap: {df['gap'].mean():.2f}")
plt.xlabel("$P^{max}$ - Markov accuracy")
plt.ylabel("Number of Users")
plt.title("Predictability left on the table by the simple Markov baseline")
plt.legend()
plt.tight_layout()
plt.show()
