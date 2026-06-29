import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import tqdm
import re

MAIN_DIR = Path(__file__).parent.parent.parent
IMPUT_FILES =  MAIN_DIR /  f"Database/no_duplicate"
IMPUT_DIR = MAIN_DIR / f"results/numpy"
OUTPUT_DIR = MAIN_DIR / f"results/numpy"

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

def get_day(filepath : Path) -> str:
    return filepath.name.split("_")[0]

transition_matrixs = [np.load(IMPUT_DIR / f"transition_matrix_normalized_{merge_function}.npy", allow_pickle=True).item() for merge_function in MERGE]
files = [file for file in IMPUT_FILES.glob("*.csv")]


def entropy(probabilities):
    sum_entropy = 0
    for p in probabilities:
        if p > 0:
            sum_entropy += -p * np.log2(p)
    return sum_entropy

def calculate_transition_entropies(transition_matrixs):
    transition_entropy = [{} for _ in transition_matrixs]
    for i,transition_matrix in enumerate(transition_matrixs):
        for cell_from in transition_matrix:
            probabilities = list(transition_matrix[cell_from].values())
            transition_entropy[i][cell_from] = entropy(probabilities)
        np.save(OUTPUT_DIR / f"transition_entropy_{MERGE[i]}.npy", transition_entropy)

    return transition_entropy

def entropy_cells_by_user(user_cells, user_stamps):
    total_transitions = len(user_cells) - 1
    cells = []
    for c in user_cells:
        if not c in cells:
            cells.append(c)

    if total_transitions <= 0:
        return 0
    
    # We create the transition matrix for the user
    
    mat_transitions = {cell_from:{cell_to:0 for cell_to in cells} for cell_from in cells}
    mat_transitions['outside'] = {cell_to:0 for cell_to in cells}
    for c in mat_transitions:
        mat_transitions[c]['outside'] = 0

    for i in range(total_transitions):
        if user_stamps[i + 1] - user_stamps[i] <= 4 * 3600 + 60:
            cell_from = user_cells[i]
            cell_to = user_cells[i + 1]
            mat_transitions[cell_from][cell_to] += 1
        else:
            cell_out = user_cells[i]
            cell_in = user_cells[i + 1]
            mat_transitions[cell_out]['outside'] += 1
            mat_transitions['outside'][cell_in] += 1

    # Normalize the matrix

    for cell_from in mat_transitions:
        for cell_to in mat_transitions:
            mat_transitions[cell_from][cell_to] /= total_transitions


    # Calcul of the entropy and relative entropy for the user

    entropy_cells = {}
    entropy_relative = {}
    for cell_from in cells:
        probabilities = list(mat_transitions[cell_from].values())
        entropy_cells[cell_from] = entropy(probabilities)
        entropy_relative[cell_from] = entropy(probabilities)/total_transitions
    

    return entropy_cells,entropy_relative

def plot_entropy_cells_by_user(dic_entropy):
    cells = dic_entropy.keys()
    vals = dic_entropy.values()
    plt.bar(cells,vals)
    plt.xlabel("Cells")
    plt.ylabel("Entropy")
    plt.tight_layout()
    plt.show()

def entropy_for_user(user_cells, user_stamps):
    total_transitions = len(user_cells) - 1
    cells = []
    for c in user_cells:
        if not c in cells:
            cells.append(c)
    cells.append('outside')

    if total_transitions <= 0:
        return 0,0

    nb_gap = 0
    for i in range(total_transitions):
        if user_stamps[i+1] - user_stamps[i] > 4 * 3600 + 60:
            nb_gap+=1

    entropy_user= 0

    for cell in cells:
        pi = 0
        if cell == 'outside':
            pi = nb_gap / (total_transitions + 1 + nb_gap)
        else:
            pi = user_cells.count(cell)/(total_transitions + 1 + nb_gap)

        if pi != 0:
            entropy_user -= pi * np.log2(pi)

    entropy_user_rel = entropy_user / len(cells)

    return (entropy_user, entropy_user_rel)


def user_entropy_by_number_of_records(merge_function="no_merge"):
    """
    The aim of this function is to calculate the entropy of all user and classify them by the number of records of the user
    
    return nothing
    save the results in numpy file
    """

    df = {}

    for file in tqdm.tqdm(files):
        day = get_day(file)
        with open(file, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=';')
            for line in reader:
                id_user = line[0]
                if merge_function in MERGE:
                    user_cells = [MERGE[merge_function](c) for c in line[8::2] if c]
                else:
                    user_cells = [c for c in line[8::2] if c]

                user_stamps = [int(ts) for ts in line[9::2] if ts]

                user_entropy, user_relative_entropy = entropy_for_user(user_cells, user_stamps)

                df[id_user] = (day,user_entropy, user_relative_entropy,len(user_cells))

    np.save(OUTPUT_DIR / f"user_entropies_{merge_function}.npy", df)




def cells_and_stamps(id_utilisateur,idice_folder):


    with open(files[idice_folder], mode='r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            if int(line[0]) == id_utilisateur:
                utilisateur = line
                break

    user_cells = [cell for cell in utilisateur[8::2]]
    user_stamps = [int(ts) for ts in utilisateur[9::2]]

    return user_cells,user_stamps

if __name__ == "__main__":
    #transition_entropy = calculate_transition_entropies()

    #for i,merge in enumerate(MERGE):
        #print("Transition entropy for ", merge, " mean: ", np.mean(list(transition_entropy[i].values())))
        #print("Transition entropy for ", merge, " std: ", np.std(list(transition_entropy[i].values())))
        #print("Transition entropy for ", merge, " min: ", np.min(list(transition_entropy[i].values())), "for cell: ", min(transition_entropy[i], key=transition_entropy[i].get))
        #print("Transition entropy for ", merge, " max: ", np.max(list(transition_entropy[i].values())), "for cell: ", max(transition_entropy[i], key=transition_entropy[i].get))

    #id_utilisateur,idice_folder = 1277,0
    #user_cells,user_stamps = cells_and_stamps(id_utilisateur,idice_folder)
    #entropy_user,entropy_relative = entropy_for_user(user_cells, user_stamps)
    #print(entropy_user,entropy_relative)

    user_entropy_by_number_of_records()