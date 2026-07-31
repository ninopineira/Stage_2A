from pathlib import Path
import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes

import pandas as pd

MAIN_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = MAIN_DIR / f"results/predictions/simple_predictor/transition_matrix/training_matrix"
dict_paths = [OUTPUT_DIR / "ngrams_matrix_train_random.json", OUTPUT_DIR / "limited_ngrams_matrix_train_random.json"]
CONTEXT_LENGTH = [1,2,3,4,5]

# =======================
# STATS                 
# =======================

fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, figsize=(20,30))
axes : np.ndarray[Axes]

fig2, axes2 = plt.subplots(nrows=5, ncols=2, sharex=True, figsize=(20,30))
axes2 : np.ndarray[Axes]


for i,path in enumerate(dict_paths):
    with open(path, mode='r') as f:
        data = json.load(f)

    output_stats_dir = OUTPUT_DIR / "stats"
    output_stats_dir.mkdir(parents=True, exist_ok=True)
    for gram in tqdm.tqdm(CONTEXT_LENGTH,desc=f"computing stats..."):
        s = 0
        number_suffix = []
        only_one_suffix_counter = 0
        for context_string,suffix_counter in data[str(gram)].items():
            n_suffix = len(suffix_counter)
            s += n_suffix
            number_suffix.append(n_suffix)
            if n_suffix == 1:
                only_one_suffix_counter += 1

        
        stats = pd.Series(number_suffix)
        stats_described = stats.describe()
        # Count     : nombre de contextes différents (car 1 longueur de suffixe par contexte)
        # mean      : nbre de suffixe moyen par contexte
        # std       : écart-type de suffixe moyen par contexte
        # min       : nbre de suffixe min parmis tous les contextes (si c'est pas 1 je me coupe une couille)
        # 25,50,75% : quartiles
        # max       : plus grand nombre de suffixe existant parmis tous les contextes

        stats_described["one_suffix_count"] = only_one_suffix_counter

        output_stats = output_stats_dir /  (path.stem + f"_stats_gram_{gram}.csv")
        stats_described.to_csv(path_or_buf=output_stats)
        
        if i <= 1:
            ax = axes[gram-1][i]
            ax : Axes
            ax.hist(x = number_suffix, bins='auto')
            ax.set_yscale(value="log")
            if i%2 == 0:
                ax.set_title(f"Number of suffix distribution\nContext length = {gram} | Mode = Full")
            else:
                ax.set_title(f"Number of suffix distribution\nContext length = {gram} | Mode = Limited")
            ax.set_xlabel("Number of suffix")
            ax.set_ylabel("Number of different context")
        else:
            ax = axes2[gram-1][i-2]
            ax : Axes
            ax.hist(x = number_suffix, bins='auto')
            ax.set_yscale(value="log")
            if i%2 == 0:
                ax.set_title(f"Number of suffix distribution\nContext length = {gram} | Mode = Full")
            else:
                ax.set_title(f"Number of suffix distribution\nContext length = {gram} | Mode = Limited")
            ax.set_xlabel("Number of suffix")
            ax.set_ylabel("Number of different context")
            
fig.suptitle("Distribution of the number of suffixes for the full dataset")
fig2.suptitle("Distribution of the number of suffixes for train data")

fig.tight_layout()
fig2.tight_layout()

fig.savefig(output_stats_dir / f"dist_number_of_suffix_full.png", dpi=300)
fig2.savefig(output_stats_dir / f"dist_number_of_suffix_train.png", dpi=300)