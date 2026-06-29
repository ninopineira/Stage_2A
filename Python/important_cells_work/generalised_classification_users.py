import csv
import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from utils import get_day

"""
The aim of this script is to classify users based on their presence at base stations during the day.

To do this, we will use a 6-bit binary classification, with each bit corresponding to a 4-hour time slot (0am-4am, 4am-8am, 8am-12pm, 12pm-4pm, 4pm-8pm, 8 pm-midnight), 
if the bit is set to 1, the user is considered to have been present at at least one base station during that time slot; otherwise, they are considered absent.

There are therefore 64 possible classes, which will provide us with more precise information on user presence in the study area
 """


MAIN_DIR = Path(__file__).parent.parent.parent

INPUT_DIR = MAIN_DIR /  f"Database/no_duplicate"

OUTPUT_DIR = MAIN_DIR / f"results/intermediate_result"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

files = [file for file in INPUT_DIR.glob("*.csv")]

result_day = {"user_id" : [],
             "day" : [],
             
             "user_presence_classification" : []
}

result_day_bis = {"user_id" : [],
             "day" : [],
             
             "user_presence_classification" : [], # cette liste comporteras un code binaire codé sur 6 bits, chaque bit corresponds à une tranche de 4h (0h-4h, 4h-8h, 8h-12h, 12h-16h, 16h-20h, 20h-24h), si le bit est à 1 alors l'utilisateur a été considéré comme présent dans au moins une station de base pendant cette tranche horaire, sinon il a été considéré comme absent
             "user_presence_cluster" : [], # cette liste comporteras un nombre entre 0 et 4, correspondant au cluster auquel appartient l'utilisateur, qui correspondra à son profil d'utilisation
             "user_presence_cluster_name" : [] # cette liste comporteras le nom du cluster auquel
             }

window = {"00h-04h" : (0,14400),
          "04h-08h" : (14400,28800),
          "08h-12h" : (28800,43200),
          "12h-16h" : (43200,57600),
          "16h-20h" : (57600,72000),
          "20h-24h" : (72000,86400)
          }

# ======================== #
# CLASSIFICATION FUNCTIONS #
# ======================== #

def classify(stamps : list[int]):
    """
    This function will create the binary code

    cells : a list of cell ids where the user was present during the day
    stamps : a list of timestamps corresponding to the times when the user was connected at the base stations

    return : a number between 0 and 63, corresponding to the binary code of the user's presence during the day
    """

    presence = [0,0,0,0,0,0] # we initialize the presence list with 6 bits set to 0, which means that the user is considered absent during all time slots

    for stamp in stamps:
        for i, (start, end) in enumerate(window.values()):
            if start <= stamp < end: # if the timestamp is within the time slot, we set the corresponding bit to 1, which means that the user is considered present during this time slot
                presence[i] = 1
    
    presence_code = ''
    for i, bit in enumerate(presence):
        presence_code += str(bit) # we convert the binary code to a string, which will be easier to manipulate and store in the dataframe

    return presence_code

def a_gap(user_stamps : list[int]):
    """
    This function will determine if the user has a gap during the day, which means that he is present at base stations during the day, but with a gap during the day (between 4am and 8pm)

    user_stamps : a list of timestamps corresponding to the times when the user was connected at the base stations

    return : True if the user has a gap during the day, False otherwise
    """
    previous_stamp = user_stamps[0]

    for stamp in user_stamps[1:]:
        if stamp - previous_stamp > 4*3600 + 60: # if the timestamp is between 4am and 8pm, we consider that the user has a gap during the day
            return True
        previous_stamp = stamp
    
    return False



def classify_profil(user_stamps : list[int]):
    """
    This function will classify the user based on his presence at base stations during the day, using the 5 clusters defined above

    user_stamps : a list of timestamps corresponding to the times when the user was connected at the base stations

    return : a number between 0 and 4, corresponding to the cluster to which the user belongs, which will correspond to the user profile
    return : None if the user profil not defined
    """
    if len(user_stamps) == 0:
        return None # if the user has no presence at base stations during the day, we consider that he belongs to cluster 4 (users who are present at base stations during the day, but not during the morning(before 4am) and the evening (after 8pm))
    
    has_gap = a_gap(user_stamps)

    morning_presence =  user_stamps[0] < 14400 
    evening_presence = user_stamps[-1] >= 72000

    if morning_presence and evening_presence and not has_gap:
        return 0 # users who are present at base stations during all time periods
    
    elif morning_presence and evening_presence and has_gap:
        return 1 # users who are present at base stations during the day, but with a gap during the day (between 4am and 8pm)
    
    elif morning_presence and not evening_presence:
        return 2 # users who are present at the morning (before 4am), but they leave the area during the day
    
    elif not morning_presence and evening_presence:
        return 3 # users who are present at the evening (after 8pm), and they arrive in the area during the day
    
    elif not morning_presence and not evening_presence:
        return 4 # users who are present at base stations during the day, but not during the morning(before 4am) and the evening (after 8pm)
    
    else:
        return None # if the user profil not defined
    
def name_cluster(cluster_code : int):
    if not 0 <= cluster_code <= 4:
        return "Unknown"
    
    cluster_names = {
        0 : "Always present",
        1 : "Home but out workers",
        2 : "Night residents leaving during the day",
        3 : "Day arrivals staying overnight",
        4 : "Passing through the area"
    }

    return cluster_names.get(cluster_code, "Unknown")

# ========================= #
# CLASSIFICATION PROCESSING #
# ========================= #

for file in tqdm.tqdm(files):
    day = get_day(file)
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter=";")
        for line in reader:
            user_id = line[0]
            user_cells = [c for c in line[8::2] if c]
            user_stamps = [int(ts) for ts in line[9::2] if ts]

            presence_code = classify(user_stamps)

            presence_cluster = classify_profil(user_stamps)
            presence_cluster_name = name_cluster(presence_cluster)

            result_day["user_id"].append(user_id)
            result_day["day"].append(day)
            result_day["user_presence_classification"].append(presence_code)

            result_day_bis["user_id"].append(user_id)
            result_day_bis["day"].append(day)
            result_day_bis["user_presence_classification"].append(presence_code)
            result_day_bis["user_presence_cluster"].append(presence_cluster)
            result_day_bis["user_presence_cluster_name"].append(presence_cluster_name)

            

df = pd.DataFrame(result_day)
df_bis = pd.DataFrame(result_day_bis)


df = df.groupby(by=["day","user_presence_classification"]).size().unstack(fill_value=0)

df.to_csv(OUTPUT_DIR / "user_generalised_classification.csv", sep=";", header=True, index=True, index_label=True)
df_bis.to_csv(OUTPUT_DIR / "user_generalised_classification_by_user.csv", sep=";", header=True, index=False)



