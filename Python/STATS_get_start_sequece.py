from collections import Counter
import csv
import json
from pathlib import Path
from typing import Counter
import tqdm
from utils import get_day
import re
import pandas as pd

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

def most_cell_present(cells: list[str], stamps: list[int], period: tuple[int, int]) -> str:
    """Determine the most present cell in a given period.
    
    cells : a list of cell ids where the user was present during the day
    stamp : a list of timestamps corresponding to the times when the user was connected at the base
    station period : a tuple of two integers representing the start and end time of the period (in seconds since the beginning of the day)

    return : the cell id of the most present cell during the period
    """
    time_stayed_in_cells = Counter()
    old_cell = cells[0]
    old_stamp = stamps[0]
    for current_cell,current_timestamp in zip(cells[1:], stamps[1:]):
        if current_cell != old_cell:
            if current_timestamp-old_stamp < 4*3600: # if the time between two records is more than 4 hours, we consider that the user has disconnected and reconnected and we reset the time stayed in cells
                time_stayed_in_cells[old_cell] += current_timestamp-old_stamp
                old_cell = current_cell
                old_stamp = current_timestamp
                    
            else:
                old_cell = current_cell
                old_stamp = current_timestamp
            
        else: # if the user is still in the same cell, we just update the time stayed in this cell
            if current_timestamp-old_stamp < 4*3600: # if the time between two records is more than 4 hours, we consider that the user has disconnected and reconnected and we reset the time stayed in cells
                time_stayed_in_cells[old_cell] += current_timestamp-old_stamp
                old_cell = current_cell
                old_stamp = current_timestamp
                    
            else:
                old_cell = current_cell
                old_stamp = current_timestamp
    
    return time_stayed_in_cells.most_common(1)[0][0] if time_stayed_in_cells else None


def entree_exit(line, morning, evening, activity_time = (5*3600, 19*3600), merge_function = None):

    if merge_function is None:
        user_cells = [c for c in line[8::2]]
        user_stamps = [int(ts) for ts in line[9::2]]

    else:
        user_cells = [MERGE[merge_function](c) for c in line[8::2]]
        user_stamps = [int(ts) for ts in line[9::2]]

    Home_morning = most_cell_present(user_cells, user_stamps, (0, morning))
    Home_evening = most_cell_present(user_cells, user_stamps, (evening, 24*3600))

    if len(user_cells) <= 1 or len(user_stamps) <= 1:
        return None, None

    start_time, end_time = activity_time

    entrances = []
    exits = []
    previous_stamp = user_stamps[0]
    previous_cell = user_cells[0]

    for cell,stamp in zip(user_cells, user_stamps):
        if previous_stamp < start_time and stamp >= start_time:
            if cell != Home_morning:
                entrances.append(cell)
        
        if previous_stamp <= end_time and stamp > end_time:
            if previous_cell != Home_evening:
                exits.append(previous_cell)
        
        if stamp >= start_time and stamp <= end_time:
            if stamp/3600 - previous_stamp/3600 > 4: # if the time between two connections is more than 4 hours, we consider that the user has disconnected and reconnected
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

def build_day_rows(day, list_entrance, list_exit):
    all_stations = set(list_entrance) | set(list_exit)
    rows = []
    for station in all_stations:
        rows.append({
            "day": day,
            "station": station,
            "entrance_count": occurences(station, list_entrance),
            "exit_count": occurences(station, list_exit),
        })
    return rows


rows_no_merge = []
rows_simple = []
rows_2g3g = []

for file in tqdm.tqdm(files):

    list_entrance = []
    list_exit = []
    list_entrance_simple_merged = []
    list_exit_simple_merged = []
    list_entrance_2g3g_merged = []
    list_exit_2g3g_merged = []

    day = get_day(file)
    with open(file, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            entrance, exits = entree_exit(line,4*3600,2*3600, merge_function=None)
            if entrance is None or exits is None:
                break

            list_entrance += entrance
            list_exit += exits

            entrance_simple_merged, exit_simple_merged = entree_exit(line,4*3600,2*3600, merge_function="simple")
            list_entrance_simple_merged += entrance_simple_merged
            list_exit_simple_merged += exit_simple_merged

            entrance_2g3g_merged, exit_2g3g_merged = entree_exit(line,4*3600,2*3600, merge_function="2g3g")
            list_entrance_2g3g_merged += entrance_2g3g_merged
            list_exit_2g3g_merged += exit_2g3g_merged

    rows_no_merge += build_day_rows(day, list_entrance, list_exit)
    rows_simple += build_day_rows(day, list_entrance_simple_merged, list_exit_simple_merged)
    rows_2g3g += build_day_rows(day, list_entrance_2g3g_merged, list_exit_2g3g_merged)


pd.DataFrame(rows_no_merge).sort_values(["day", "station"]).to_csv(INTERMEDIATE_PATH / "stats_entrance_exit_no_merge.csv", index=False, sep=";")
pd.DataFrame(rows_simple).sort_values(["day", "station"]).to_csv(INTERMEDIATE_PATH / "stats_entrance_exit_simple_merge.csv", index=False, sep=";")
pd.DataFrame(rows_2g3g).sort_values(["day", "station"]).to_csv(INTERMEDIATE_PATH / "stats_entrance_exit_2g3g_merge.csv", index=False, sep=";")