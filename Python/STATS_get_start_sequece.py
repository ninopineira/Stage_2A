import csv
import json
from pathlib import Path
import utils
import tqdm
from utils import get_day
import matplotlib.pyplot as plt

MAIN_DIR = Path(__file__).parent.parent
INPUT_DIR = MAIN_DIR /  f"Database/no_duplicate" 
files = [path for path in INPUT_DIR.glob("*.csv")]

INTERMEDIATE_PATH = MAIN_DIR / f"results/intermediate_result"


# L'objectif de ce script est de determiner les proportions des stations de base qui sont des entrées.

def entree_exit(line):

    user_cells = [c for c in line[8::2] if c]
    user_stamps = [int(ts) for ts in line[9::2] if ts]

    entree = []
    exits = []
    previous_stamp = user_stamps[0]
    previous_cell = user_cells[0]

    for cell,stamp in zip(user_cells, user_stamps):
        if stamp/3600 - previous_stamp/3600 > 4 and 5*3600 < stamp/3600 < 19*3600: # if the time between two connections is more than 4 hours, we consider that the user has disconnected and reconnected
            entree.append(cell)
            exits.append(previous_cell)

    return entree, exits

def occurences(station, list_stations):
    return list_stations.count(station)





# ============================ #
# LEAGING AREA STATS FUNCTIONS #
# ============================ #
# TODO



for file in tqdm.tqdm(files):
    
    list_entree = []
    list_exit = []
    day = get_day(file)
    with open(file, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            entree, exits = entree_exit(line)

            list_entree += entree
            list_exit += exits

    dict_entree = {}
    dict_exit = {}

    for station in set(list_entree):
        dict_entree[station] = occurences(station, list_entree)

    for station in set(list_exit):
        dict_exit[station] = occurences(station, list_exit)

    print(f"Day {day} : {len(list_entree)} entrées et {len(list_exit)} sorties")
    print(f"Nombre de stations de base différentes : {len(set(list_entree + list_exit))}")
