import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import re
import numpy as np

MAIN_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = MAIN_DIR / f"results/intermediate_result"

files = [file for file in INPUT_DIR.glob("cell_activity*.csv")]

colors = {'1': "blue", '2': "orange"}

for file in files:

    df = pd.read_csv(file, sep=";", header=[0, 1], index_col=0)

    # delete the row with index "True" if it exists
    df = df.drop(index="True", errors="ignore")

    # Convert all values to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Get the unique periods from the first level of the column MultiIndex
    periods = df.columns.get_level_values(0).unique()

    n_periods = len(periods)
    x = np.arange(len(df))
    width = 0.8 / n_periods

    for i, period in enumerate(periods):
        sub = df[period]  # DataFrame avec colonnes 1 et 2
        bottom = np.zeros(len(df))
        for col in sub.columns:
            ax.bar(x + i * width, sub[col], width, bottom=bottom, color=colors[col],edgecolor="white")
            bottom += sub[col].values

    ax.set_xticks(x + width * n_periods / 2)
    ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.set_title(f"Number of {file.stem}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Number of person")
    plt.tight_layout()

    plt.savefig(MAIN_DIR / f"results/plots/distribution_{file.stem}.png")
    plt.show()