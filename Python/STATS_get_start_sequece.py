from collections import Counter
import csv
import json
from pathlib import Path
from typing import Counter
import tqdm
from utils import get_day
import re
import pandas as pd
from important_cells_work.generalised_classification_users import classify_profil

MAIN_DIR = Path(__file__).parent.parent
INPUT_DIR = MAIN_DIR /  f"Database/no_duplicate"
files = [path for path in INPUT_DIR.glob("*.csv")]

INTERMEDIATE_PATH = MAIN_DIR / f"results/intermediate_result"
INTERMEDIATE_PATH.mkdir(parents=True, exist_ok=True)


# L'objectif de ce script est de determiner les proportions des stations de base qui sont des entrées.

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
    "simple" : get_cell_code,
    "2g3g" : get_cell_code2
}


def entree_exit(line, morning, evening, merge_function = None):

    if merge_function is None:
        user_cells = [c for c in line[8::2]]
        user_stamps = [int(ts) for ts in line[9::2]]

    else:
        user_cells = [MERGE[merge_function](c) for c in line[8::2]]
        user_stamps = [int(ts) for ts in line[9::2]]

    if len(user_cells) <= 1 or len(user_stamps) <= 1:
        return [],[]


    entrances = []
    exits = []
    previous_stamp = user_stamps[0]
    previous_cell = user_cells[0]

    if user_stamps[0] > morning:
        entrances.append(user_cells[0])

    if user_stamps[-1] < evening:
        exits.append(user_cells[-1])

    for cell,stamp in zip(user_cells, user_stamps):
        if stamp - previous_stamp > 4*3600 + 60: # if the time between two connections is more than 4 hours, we consider that the user has disconnected and reconnected
            entrances.append(cell)
            exits.append(previous_cell)
        previous_stamp = stamp
        previous_cell = cell

    return entrances, exits


def occurences(station, list_stations):
    return list_stations.count(station)





# ============================ #
# LEAGING AREA STATS FUNCTIONS #
# ============================ #
# TODO

def build_day_rows(day, list_entrance, list_exit, cluster_type):
    all_stations = set(list_entrance) | set(list_exit)
    rows = []
    for station in all_stations:
        rows.append({
            "day": day,
            "station": station,
            "entrance_count": occurences(station, list_entrance),
            "exit_count": occurences(station, list_exit),
            "cluster_type": cluster_type  # Placeholder for user profile type
        })
    return rows


rows_no_merge = []
rows_simple = []
rows_2g3g = []

for file in tqdm.tqdm(files):

    list_entrance = {i:[] for i in range(5)}
    list_exit = {i:[] for i in range(5)}
    list_entrance_simple_merged = {i:[] for i in range(5)}
    list_exit_simple_merged = {i:[] for i in range(5)}
    list_entrance_2g3g_merged = {i:[] for i in range(5)}
    list_exit_2g3g_merged = {i:[] for i in range(5)}

    day = get_day(file)
    with open(file, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            user_stamps = [int(x) for x in line[9::2]]
            i = classify_profil(user_stamps)

            entrance, exits = entree_exit(line,4*3600,20*3600, merge_function=None)
            
            list_entrance[i] += entrance
            list_exit[i] += exits

            entrance_simple_merged, exit_simple_merged = entree_exit(line,4*3600,20*3600, merge_function="simple")
            list_entrance_simple_merged[i] += entrance_simple_merged
            list_exit_simple_merged[i] += exit_simple_merged

            entrance_2g3g_merged, exit_2g3g_merged = entree_exit(line,4*3600,20*3600, merge_function="2g3g")
            list_entrance_2g3g_merged[i] += entrance_2g3g_merged
            list_exit_2g3g_merged[i] += exit_2g3g_merged
            
    for i in range(5):
        rows_no_merge += build_day_rows(day, list_entrance[i], list_exit[i],i)
        rows_simple += build_day_rows(day, list_entrance_simple_merged[i], list_exit_simple_merged[i],i)
        rows_2g3g += build_day_rows(day, list_entrance_2g3g_merged[i], list_exit_2g3g_merged[i],i)


pd.DataFrame(rows_no_merge).sort_values(["day", "station","cluster_type"]).to_csv(INTERMEDIATE_PATH / "stats_entrance_exit_no_merge.csv", index=False, sep=";")
pd.DataFrame(rows_simple).sort_values(["day", "station","cluster_type"]).to_csv(INTERMEDIATE_PATH / "stats_entrance_exit_simple_merge.csv", index=False, sep=";")
pd.DataFrame(rows_2g3g).sort_values(["day", "station","cluster_type"]).to_csv(INTERMEDIATE_PATH / "stats_entrance_exit_2g3g_merge.csv", index=False, sep=";")