import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

MAIN_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = MAIN_DIR / f"results/intermediate_result"

file = INPUT_DIR / "user_generalised_classification.csv"

def top5_par_jour(row):
    top = row.nlargest(5)
    return top.reindex(row.index, fill_value=0)

if file.exists():

    df = pd.read_csv(file, sep=";", index_col=0)
    top5 = df.apply(top5_par_jour, axis=1)
    top5 = top5.loc[:, (top5 != 0).any(axis=0)]

    top5.plot(kind="bar", stacked=True, figsize=(10,6))
    plt.title("Classification of users by presence during the day")
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Day")
    plt.ylabel("Number of Users")
    plt.legend(title="Classes present", bbox_to_anchor=(1.0, 1), loc='upper left')
    
    #plt.show()

    plt.savefig(MAIN_DIR / f"results/plots/classification_by_presence.png", bbox_inches='tight')

    plt.close()
    file.unlink() # we delete the file after processing it, since it is an intermediate result that is not needed anymore
    
else:
    print("Input file not found.")