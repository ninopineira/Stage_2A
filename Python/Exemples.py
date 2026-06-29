import matplotlib.pyplot as plt
import csv
import numpy as np
from pathlib import Path
from STATS_get_start_sequece import entree_exit
import tqdm
from utils import get_day
import pandas as pd
import re

MAIN_DIR = Path(__file__).parent.parent
INPUT_DIR = MAIN_DIR /  f"Database/no_duplicate"
INTERMEDIATE_PATH = MAIN_DIR / f"results/intermediate_result"
INTERMEDIATE_PATH.mkdir(parents=True, exist_ok=True)

files = [file for file in INPUT_DIR.glob("*.csv")]

def plot_presence_over_time(unser_stamps, id_user_id, day):
    """
    Plot the presence of a user, the number of connections hour by hour in a histogram.
    """

    hourly_counts = [0] * 24
    for stamp in unser_stamps:
        hour = stamp // 3600
        hourly_counts[hour] += 1

    plt.figure(figsize=(10, 6))
    plt.bar(range(24), hourly_counts)
    plt.title(f"Presence of User {id_user_id} Over Time - {day}")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Number of Connections")
    plt.xticks(range(24))
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# 856,0 un utilisateur qui fait un passage dans la zone en journée, il a une entrée et une sortie.
# 1277,0 une utilisateur avec plein de connexions, présent tout la journée donc ni entrée ni sortie.
# 2578,0 un utilisateur présent presque toute la journée, mais qui à une absence au milieu.
# 3740,0 un utilisateur présent que très tot le matin donc pas d'entrée mais une sortie.
# 2069758,2 un utilisateur qui arrive au court de la journée mais qui reste jusqu'à la fin, il a une entrée mais pas de sortie.
# 2069878,2 un utilisateur qui n'a pas d'activité mais des pings de verification toute la journée, il n'a ni entrée ni sortie.

# 2069871,2 un exemple d'utilisateur avec des trous de partout pour montrer comment se fait l'occupation des cellules au cours de la journée.
# 992,0 un exemple d'utilisateur pour montrer le nombre de connexions par heure et par cellule.

id_utilisateur,idice_folder = 856,0

with open(files[idice_folder], mode='r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f, delimiter=';')
    for line in reader:
        if int(line[0]) == id_utilisateur:
            utilisateur = line
            break

user_stamps = [int(ts) for ts in utilisateur[9::2]]


def navigate_lines(filepath):
    with open(filepath, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        lines = list(reader)

    index = [5]
    fig, ax = plt.subplots(figsize=(10, 6))

    def draw(idx):
        line = lines[idx]
        user_stamps = [int(ts) for ts in line[9::2]]

        hourly_counts = [0] * 24
        for stamp in user_stamps:
            hourly_counts[stamp // 3600] += 1

        ax.clear()
        ax.bar(range(24), hourly_counts)
        ax.set_title(f"Ligne {idx + 1}/{len(lines)} — Utilisateur {line[0]}")
        ax.set_xlabel("Heure de la journée")
        ax.set_ylabel("Nombre de connexions")
        ax.set_xticks(range(24))
        ax.grid(axis='y', alpha=0.75)
        fig.canvas.draw()

    def on_key(event):
        if event.key == 'right':
            index[0] = (index[0] + 1) % len(lines)
            draw(index[0])
        elif event.key == 'left':
            index[0] = (index[0] - 1) % len(lines)
            draw(index[0])
        elif event.key == ' ':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw(index[0] % len(lines))
    plt.show()



def presence_in_cells(cells,stamps,dic_cells_nb_user,dic_cells_nb_connections):
    """
    This function will create a dictionary with the frequency of use of each cell by hour

    cells : a list of cell ids where the user was present during the day
    stamps : a list of timestamps corresponding to the times when the user was connected at the base stations
    dic_cells_nb_user : a dictionary with the cell ids as keys and a list of 24 integers which represent the number of users present at each hour as values
    dic_cells_nb_connections : a dictionary with the cell ids as keys and a list of 24 integers which represent the number of connections at each hour as values

    return : a dictionary with the cell ids as keys and a list of 24 intergers which represent the number of times the user was present at each hour as values
    and a dictionary with the cell ids as keys and a list of 24 intergers which represent the number of times the user was connected at each hour as values
    """
    if len(cells) == 0 or len(stamps) == 0:
        return dic_cells_nb_user, dic_cells_nb_connections

    previous_cell = cells[0]
    previous_stamp = stamps[0]

    passed_hours = {cell: [False]*24 for cell in dic_cells_nb_user} # we create a dictionary to keep track of the hours that have already been counted for the current cell

    # morning and evening cases
    firtst_stamp = stamps[0]
    last_stamp = stamps[-1]
    if firtst_stamp < 4 * 3600 : # if the first timestamp is before 4am, we consider that the user is present on the first cell until 4am
        for hour in range(0, firtst_stamp // 3600):
            dic_cells_nb_user[cells[0]][hour] += 1
            passed_hours[cells[0]][hour] = True
        
    if last_stamp >= 20 * 3600 : # if the last timestamp is after 8pm, we consider that the user is present on the last cell until 8pm
        for hour in range(last_stamp // 3600 , 24):
            dic_cells_nb_user[cells[-1]][hour] += 1
            passed_hours[cells[-1]][hour] = True

    for cell, stamp in zip(cells, stamps):
        dic_cells_nb_connections[cell][stamp // 3600] += 1
        if cell != previous_cell:
            if stamp - previous_stamp < 4*3600 + 60: # if the user is connected to a different cell within 4 hours, we consider that he is still present in the previous cell
                hour_start = previous_stamp // 3600 # we convert the timestamp to hours
                hour_end = stamp // 3600
                for hour in range(hour_start, hour_end + 1):
                    if not passed_hours[previous_cell][hour]: # if the hour has not been counted yet, we increment the frequency of use of the previous cell at the corresponding hour
                        dic_cells_nb_user[previous_cell][hour] += 1 # we increment the frequency of use of the previous cell at the corresponding hour
                        passed_hours[previous_cell][hour] = True # we mark the hour as counted
                previous_cell = cell
                previous_stamp = stamp
            else: # if the user is connected to a different cell after more than 4 hours, we consider that he is present on the previous cell until the end of the hour
                hour= previous_stamp // 3600
                if not passed_hours[previous_cell][hour]: # if the hour has not been counted yet, we increment the frequency of use of the previous cell at the corresponding hour
                    dic_cells_nb_user[previous_cell][hour] += 1
                    passed_hours[previous_cell][hour] = True
                previous_cell = cell
                previous_stamp = stamp
        else: # if the user is connected to the same cell, we consider that he is present on this cell until the end of the hour
            if stamp - previous_stamp < 4*3600 + 60: # if the user is connected to the same cell within 4 hours, we consider that he is still present in this cell
                hour_start = previous_stamp // 3600
                hour_end = stamp // 3600
                for hour in range(hour_start, hour_end):
                    if not passed_hours[cell][hour]:
                        dic_cells_nb_user[cell][hour] += 1 # we increment the frequency of use of the cell at the corresponding hour
                        passed_hours[cell][hour] = True
                previous_stamp = stamp
            else: # if the user is connected to the same cell after more than 4 hours, we consider that he is present on this cell until the end of the hour
                hour = previous_stamp // 3600
                if not passed_hours[cell][hour]:
                    dic_cells_nb_user[cell][hour] += 1
                    passed_hours[cell][hour] = True
                previous_stamp = stamp
    
    if not passed_hours[cells[-1]][last_stamp // 3600]:
        dic_cells_nb_user[cells[-1]][last_stamp // 3600] += 1
        passed_hours[cells[-1]][last_stamp // 3600] = True
    
    return dic_cells_nb_user, dic_cells_nb_connections

INPUT_CELLS = MAIN_DIR / "Database/cells/cd_142_cells.csv"

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


cells_df = pd.read_csv(INPUT_CELLS, sep=";")
cell_ids = cells_df["cellid"].tolist()

dict_cells_nb_user = {cell: [0]*24 for cell in cell_ids}
dict_cells_nb_connections = {cell: [0]*24 for cell in cell_ids}

user_cells = [c for c in utilisateur[8::2]]
user_stamps = [int(ts) for ts in utilisateur[9::2]]
dic_cells_nb_user, dic_cells_nb_connections = presence_in_cells(user_cells, user_stamps, dict_cells_nb_user, dict_cells_nb_connections)

def affichage_cells(dic):
    for cell in dic:
        if sum(dic[cell]) > 0:
            plt.figure(figsize=(10, 6))
            plt.bar(range(24), dic[cell])
            plt.title(f"Cell {cell} — User {id_utilisateur}")
            plt.xlabel("Hour of the Day")
            plt.ylabel("Number of Connections")
            plt.xticks(range(24))
            plt.grid(axis='y', alpha=0.75)
            plt.show()

def afficher_ex_class(files = INTERMEDIATE_PATH / "user_generalised_classification_by_user.csv"):
    """
    This function will sort users by cluster and show a random example of user for each cluster with the presence over time and the cells used.
    """
    df = pd.read_csv(files, sep=";")
    for cluster in range(5):
        users_in_cluster = df[df["user_presence_cluster"] == cluster]
        if not users_in_cluster.empty:
            random_user = users_in_cluster.sample(n=1).iloc[0]
            user_id = random_user["user_id"]
            day = random_user["day"]
            print(f"Cluster {cluster} - User {user_id} - Day {day}")
            user_file = INPUT_DIR / f"{day}_no_duplicate.csv"
            with open(user_file, mode='r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f, delimiter=';')
                for line in reader:
                    if int(line[0]) == user_id:
                        utilisateur = line
                        break
            user_stamps = [int(ts) for ts in utilisateur[9::2]]
            plot_presence_over_time(user_stamps, id_user_id=user_id, day=day)
            user_cells = [c for c in utilisateur[8::2]]
            print(f"Cells used by user {user_id}: {set(user_cells)}")

        


if __name__ == "__main__":
    #plot_presence_over_time(user_stamps,id_user_id=id_utilisateur, day=get_day(files[idice_folder]))

    #print("Enstrances and exits for user", id_utilisateur)
    #entrances, exits = entree_exit(utilisateur, 4*3600, 20*3600, merge_function=None)
    #print("Entrances:", entrances)
    #print("Exits:", exits)

    navigate_lines(files[idice_folder])

    affichage_cells(dic_cells_nb_user)

    afficher_ex_class()