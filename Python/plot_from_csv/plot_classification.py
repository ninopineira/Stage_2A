import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

MAIN_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = MAIN_DIR / f"results/intermediate_result"

file = INPUT_DIR / "user_generalised_classification.csv"

def top5_by_day(row):
    top = row.nlargest(5)
    return top.reindex(row.index, fill_value=0)

if file.exists():

    df = pd.read_csv(file, sep=";", index_col=0)
    top5 = df.apply(top5_by_day, axis=1)

    remains = df.sum(axis=1) - top5.sum(axis=1)
    top5["Remains"] = remains

    top5 = top5.loc[:, (top5 != 0).any(axis=0)]

    colors_20 = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#469990", "#9A6324",
        "#800000", "#aaffc3", "#808000", "#000075", "#a9a9a9",
        "#fabed4", "#dcbeff", "#fffac8", "#ffd8b1", "#333333",
    ]
    colors = colors_20[:len(top5.columns)]

    top5.plot(kind="bar", stacked=True, figsize=(10,6), color=colors)
    plt.title("Classification of users by presence during the day")

    weekends = {"15", "16", "22", "23"}
    ax = plt.gca()
    labels = [str(idx) for idx in top5.index]
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for label in ax.get_xticklabels():
        day = label.get_text().strip().split("-")[-1].split()[-1]
        if day in weekends:
            label.set_color("red")

    plt.xlabel("Day")
    plt.ylabel("Number of Users")
    plt.legend(title="Classes present", bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.tight_layout()
    

    plt.savefig(MAIN_DIR / f"results/plots/classification_by_presence.png")

    #file.unlink() # we delete the file after processing it, since it is an intermediate result that is not needed anymore
    
else:
    print("Input file not found.")


# Second classification with 5 clusters

file_bis = INPUT_DIR / "user_generalised_classification_by_user.csv"

if file_bis.exists():
    df_bis = pd.read_csv(file_bis, sep=";")
    cluster_counts = df_bis.groupby(["day", "user_presence_cluster_name"]).size().unstack(fill_value=0)

    cluster_counts.plot(kind="bar", stacked=True, figsize=(10,6), color=colors)
    plt.title("Classification of users by presence during the day (5 clusters)")

    ax = plt.gca()
    labels = [str(idx) for idx in cluster_counts.index]
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for label in ax.get_xticklabels():
        day = label.get_text().strip().split("-")[-1].split()[-1]
        if day in weekends:
            label.set_color("red")

    plt.xlabel("Day")
    plt.ylabel("Number of Users")
    plt.legend(title="Classes present", bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(MAIN_DIR / f"results/plots/classification_by_presence_5_clusters.png")