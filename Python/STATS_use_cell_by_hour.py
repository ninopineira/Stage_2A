import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import re
import csv

"""
The aim of this srcipt is to have the frequency of use of each cell by hour, 
in order to have a better understanding of the presence of users in the study area during the day, 
and to be able to classify them based on their presence at base stations during the day.
"""

MAIN_DIR = Path(__file__).parent.parent
INPUT_DIR = MAIN_DIR / f"Database/no_duplicate"
OUTPUT_DIR = MAIN_DIR / f"results/intermediate_result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = [file for file in INPUT_DIR.glob("*.csv")]

INPUT_CELLS = MAIN_DIR / "Database/cells/cd_142_cells.csv"


def presence_in_cells(cells,stamps,dic_cells):
    """
    This function will create a dictionary with the frequency of use of each cell by hour

    cells : a list of cell ids where the user was present during the day
    stamps : a list of timestamps corresponding to the times when the user was connected at the base stations
    dic_cells : a dictionary with the cell ids as keys and a list of 24 intergers which represent the number of times the user was present at each hour as values

    return : a dictionary with the cell ids as keys and a list of 24 intergers which represent the number of times the user was present at each hour as values
    """
    if len(cells) == 0 or len(stamps) == 0:
        return dic_cells
    
    previous_cell = cells[0]
    previous_stamp = stamps[0]

    for cell, stamp in zip(cells, stamps):
        if cell != previous_cell:
            if stamp - previous_stamp < 4*3600: # if the user is connected to a different cell within 4 hours, we consider that he is still present in the previous cell
                hour_start = previous_stamp // 3600 # we convert the timestamp to hours
                hour_end = stamp // 3600
                for hour in range(hour_start, hour_end + 1):
                    dic_cells[previous_cell][hour] += 1 # we increment the frequency of use of the previous cell at the corresponding hour
                previous_cell = cell
                previous_stamp = stamp
            else: # if the user is connected to a different cell after more than 4 hours, we consider that he is present on the previous cell until the end of the hour
                hour= previous_stamp // 3600
                dic_cells[previous_cell][hour] += 1
                previous_cell = cell
                previous_stamp = stamp
        else: # if the user is connected to the same cell, we consider that he is present on this cell until the end of the hour
            if stamp - previous_stamp < 4*3600: # if the user is connected to the same cell within 4 hours, we consider that he is still present in this cell
                hour_start = previous_stamp // 3600
                hour_end = stamp // 3600
                for hour in range(hour_start, hour_end):
                    dic_cells[cell][hour] += 1 # we increment the frequency of use of the cell at the corresponding hour
                previous_stamp = stamp
            else: # if the user is connected to the same cell after more than 4 hours, we consider that he is present on this cell until the end of the hour
                hour = previous_stamp // 3600
                dic_cells[cell][hour] += 1
                previous_stamp = stamp
    
    return dic_cells

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



dict_cells = { cell : [0]*24 for cell in pd.read_csv(INPUT_CELLS, sep=";")["cellid"].tolist() } # we create a dictionary with the cell ids as keys and a list of 24 intergers which represent the number of times the user was present at each hour as values, initialized to 0
dict_cells_simple_merged = { cell : [0]*24 for cell in pd.read_csv(INPUT_CELLS, sep=";")["cellid"].apply(get_cell_code).unique().tolist() } # we create a dictionary with the cell ids as keys and a list of 24 intergers which represent the number of times the user was present at each hour as values, initialized to 0
dict_cells_2g3g_merged = { cell : [0]*24 for cell in pd.read_csv(INPUT_CELLS, sep=";")["cellid"].apply(get_cell_code2).unique().tolist() } # we create a dictionary with the cell ids as keys and a list of 24 intergers which represent the number of times the user was present at each hour as values, initialized to 0


for file in files:
    with open(file, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            user_cells = [c for c in line[8::2] if c]
            user_stamps = [int(ts) for ts in line[9::2] if ts]

            dict_cells = presence_in_cells(user_cells, user_stamps, dict_cells)
            dict_cells_simple_merged = presence_in_cells([MERGE["simple"](c) for c in user_cells], user_stamps, dict_cells_simple_merged)
            dict_cells_2g3g_merged = presence_in_cells([MERGE["2g3g"](c) for c in user_cells], user_stamps, dict_cells_2g3g_merged)

df = pd.DataFrame.from_dict(dict_cells, orient='index', columns=[f"{hour}h-{hour+1}h" for hour in range(24)])
df_simple_merged = pd.DataFrame.from_dict(dict_cells_simple_merged, orient='index', columns=[f"{hour}h-{hour+1}h" for hour in range(24)])
df_2g3g_merged = pd.DataFrame.from_dict(dict_cells_2g3g_merged, orient='index', columns=[f"{hour}h-{hour+1}h" for hour in range(24)])

df.to_csv(OUTPUT_DIR / "stats_use_cell_by_hour.csv", sep=";", header=True, index=True, index_label="cellid")
df_simple_merged.to_csv(OUTPUT_DIR / "stats_use_cell_by_hour_simple_merge.csv", sep=";", header=True, index=True, index_label="cellid")
df_2g3g_merged.to_csv(OUTPUT_DIR / "stats_use_cell_by_hour_2g3g_merge.csv", sep=";", header=True, index=True, index_label="cellid")

