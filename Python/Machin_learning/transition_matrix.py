import numpy as np
import pandas as pd
from pathlib import Path
import json
import csv
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re

"""
The aim of this script is to compute the transition matrix of users between different merge (2g, 3g),
and to save the results in a numpy file.
"""

MAIN_DIR = Path(__file__).parent.parent.parent
STATS_DIR = MAIN_DIR / "Database/no_duplicate"
CELLS_DIR = MAIN_DIR / "Database/cells/cd_142_cells.csv"
OUTPUT_DIR = MAIN_DIR / "results/numpy"

files = [file for file in STATS_DIR.glob("*.csv")]

def get_cell_code(cell: str) -> str:
    """Extract base station if merge=True."""
    if cell == '': return cell
    match = re.match(r"([a-zA-Z]+)", cell)
    return match.group(1)

def get_cell_code2(cell: str) -> str:
    """Extract base station if merge=True."""
    if cell == '': return cell
    match = re.match(r"([a-zA-Z]+)", cell)
    return match.group(1)[1:]

MERGE = {
    "no_merge": lambda x: x,
    "simple": get_cell_code,
    "2g3g": get_cell_code2
}

def add_an_user_s_transitions(user_cells, user_stamps, transition_matrix):
    for i in range(len(user_cells) - 1):
        stamp_from = user_stamps[i]
        stamp_to = user_stamps[i + 1]

        if stamp_to - stamp_from < 4*3600 + 60:
            cell_from = user_cells[i]
            cell_to = user_cells[i + 1]
            transition_matrix[cell_from][cell_to] += 1  

        else:
            cell_out = user_cells[i]
            cell_in = user_cells[i + 1]
            transition_matrix[cell_out]["outside"] += 1
            transition_matrix["outside"][cell_in] += 1
            
    return transition_matrix

def create_transition_matrix(merge_function="no_merge"):
    df_cells = pd.read_csv(CELLS_DIR, sep=";")


    transition_matrix = {MERGE[merge_function](cell): {MERGE[merge_function](cell): 0 for cell in df_cells["cellid"].unique()} for cell in df_cells["cellid"].unique()}
    transition_matrix["outside"] = {MERGE[merge_function](cell): 0 for cell in df_cells["cellid"].unique()}
    for cell in df_cells["cellid"].unique():
        transition_matrix[MERGE[merge_function](cell)]["outside"] = 0
    
    return transition_matrix






def matrix(merge_function="no_merge"):
    transition_matrix = create_transition_matrix(merge_function=merge_function)

    for file in tqdm.tqdm(files):
        with open(file, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=';')
            for line in reader:
                if merge_function in MERGE:
                    user_cells = [MERGE[merge_function](c) for c in line[8::2] if c]
                else:
                    user_cells = [c for c in line[8::2] if c]
                user_stamps = [int(ts) for ts in line[9::2] if ts]

                transition_matrix = add_an_user_s_transitions(user_cells, user_stamps, transition_matrix)

    np.save(OUTPUT_DIR / f"transition_matrix_{merge_function}.npy", transition_matrix)

    return transition_matrix

# ========================
# Normalization of the transition matrix
# ========================

def normalize_transition_matrix(transition_matrix, merge_function="no_merge"):

    transition_matrix_normalized = create_transition_matrix(merge_function=merge_function)
    
    transition_matrix = np.load(OUTPUT_DIR / f"transition_matrix_{merge_function}.npy", allow_pickle=True).item()

    for cell_from in transition_matrix:
        total_transitions = sum(transition_matrix[cell_from].values())
        if total_transitions > 0:
            for cell_to in transition_matrix[cell_from]:
                transition_matrix_normalized[cell_from][cell_to] = transition_matrix[cell_from][cell_to] / total_transitions

    np.save(OUTPUT_DIR / f"transition_matrix_normalized_{merge_function}.npy", transition_matrix_normalized)
    return transition_matrix_normalized

def plot_the_matrix(transition_matrix, title):
    """
    The aim of this matrix is to plot a little part of the transition matrix.
    """
    df = pd.DataFrame(transition_matrix).fillna(0)
    df_reduced = df.iloc[-10:, -10:]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_reduced, annot=True, fmt=".2f", cmap="Blues")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.xlabel("To Cell")
    plt.ylabel("From Cell")
    plt.show()

def exemple_of_transition_matrix_with_one_user(user_id,file_id):
    df_cells = pd.read_csv(CELLS_DIR, sep=";")
    transition_matrix = {cell: {cell: 0 for cell in df_cells["cellid"].unique()} for cell in df_cells["cellid"].unique()}
    transition_matrix["outside"] = {cell: 0 for cell in df_cells["cellid"].unique()}
    for cell in df_cells["cellid"].unique():
        transition_matrix[cell]["outside"] = 0

    with open(files[file_id], mode='r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            if int(line[0]) == user_id:
                user = line
                break
        user_cells = [c for c in user[8::2] if c]
        user_stamps = [int(ts) for ts in user[9::2] if ts]

        transition_matrix = add_an_user_s_transitions(user_cells, user_stamps, transition_matrix)

    return transition_matrix

def total_transitions(transition_matrix):
    total = 0
    for cell_from in transition_matrix:
        total += sum(transition_matrix[cell_from].values())
    return total

if __name__ == "__main__":
    for merge_function in MERGE:
        transition_matrix = matrix(merge_function=merge_function)
        transition_matrix_normalized = normalize_transition_matrix(transition_matrix, merge_function=merge_function)
        plot_the_matrix(transition_matrix, "Transition Matrix (Counts)")
        plot_the_matrix(transition_matrix_normalized, "Transition Matrix (Normalized)")
        #print("Sum of all transition from cell outside: ", sum(transition_matrix["outside"].values()))

    ## Example of transition matrix with one user

    #id_utilisateur,idice_folder = 2578,0
    #transition_matrix_user = exemple_of_transition_matrix_with_one_user(id_utilisateur, idice_folder)
    #plot_the_matrix(transition_matrix_user, f"Transition Matrix for user {id_utilisateur} (file {idice_folder})")
    
    #print("Total transitions: ", total_transitions(transition_matrix_user))
