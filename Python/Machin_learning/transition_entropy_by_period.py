import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import csv
import tqdm
import re

"""
The aim of this scrip is to caluculate the entropy of each user.
But for each user it will have 3 entropy : - one for the Morning (00h -> 04h, 00h -> 05h, 00h -> 06h)
                                           - one for the Evening (18h -> 24h, 19h -> 24h, 20h -> 24h)
                                           - one for the Day (the time remainding)
"""

MAIN_DIR = Path(__file__).parent.parent.parent
IMPUT_FILES =  MAIN_DIR /  f"Database/no_duplicate"
IMPUT_DIR = MAIN_DIR / f"results/numpy"
OUTPUT_DIR = MAIN_DIR / f"results/numpy"

files = [file for file in IMPUT_FILES.glob("*.csv")]

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

Periods = {"Morning" : [(0,4),(0,5),(0,6)],
           "Evening" : [(18,24),(19,24),(20,24)]}

def separate_period(user_cells, user_stamps,morning, evening):
    end_morning = morning[1]*3600
    start_evening = evening[0]*3600

    if len(user_cells) <= 1:
        return 0,0,0

    i_end_morning = 0
    i_start_evening = 0

    # find indice of the morning and the evening
    for i in range(len(user_stamps) - 1 ):
        if user_stamps[i] < end_morning and user_stamps[i+1] > end_morning:
            i_end_morning = i
        if user_stamps[i] < start_evening and user_stamps[i+1] > start_evening:
            i_start_evening = i

    if i_start_evening == 0:
        i_start_evening = len(user_stamps) - 1

    user_cells_morning,user_stamps_morning = user_cells[:i_end_morning+1],user_stamps[:i_end_morning+1]
    user_cells_day,user_stamps_day = user_cells[i_end_morning+1:i_start_evening+1],user_stamps[i_end_morning+1:i_start_evening+1]
    user_cells_evening,user_stamps_evening = user_cells[i_start_evening+1:],user_stamps[i_start_evening+1:]

    return entropy_for_user(user_cells_morning,user_stamps_morning), \
           entropy_for_user(user_cells_day,user_stamps_day), \
           entropy_for_user(user_cells_evening,user_stamps_evening)

if __name__ == "__main__":
    dfs = {merge_name: {} for merge_name in MERGE}

    for file in tqdm.tqdm(files):
        day = get_day(file)
        with open(file, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=';')
            for line in reader:
                id_user = line[0]
                for merge_name in MERGE:
                    user_cells = [MERGE[merge_name](c) for c in line[8::2] if c]
                    user_stamps = [int(ts) for ts in line[9::2] if ts]

                    entropies, relative_entropies = [], []

                    for morning in Periods["Morning"]:
                        for evening in Periods["Evening"]:
                            e_morn, e_day, e_even = separate_period(user_cells, user_stamps, morning, evening)
                            for e in (e_morn, e_day, e_even):
                                if e == 0:
                                    entropies.append(0.0)
                                    relative_entropies.append(0.0)
                                else:
                                    entropies.append(e[0])
                                    relative_entropies.append(e[1])

                    if day not in dfs[merge_name]:
                        dfs[merge_name][day] = []
                    dfs[merge_name][day].append(entropies + relative_entropies + [float(len(user_cells))])

    for merge_name in MERGE:
        np.save(OUTPUT_DIR / f"user_entropies_by_period_{merge_name}.npy", dfs[merge_name])
