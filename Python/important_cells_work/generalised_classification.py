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
             
             "user_presence_classification" : [] # cette liste comporteras un code binaire codé sur 6 bits, chaque bit corresponds à une tranche de 4h (0h-4h, 4h-8h, 8h-12h, 12h-16h, 16h-20h, 20h-24h), si le bit est à 1 alors l'utilisateur a été considéré comme présent dans au moins une station de base pendant cette tranche horaire, sinon il a été considéré comme absent
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

def classify(cells : list[str], stamps : list[int]):
    """
    This function will create the binary code

    cells : a list of cell ids where the user was present during the day
    stamps : a list of timestamps corresponding to the times when the user was connected at the base stations

    return : a number between 0 and 63, corresponding to the binary code of the user's presence during the day
    """

    presence = [0,0,0,0,0,0] # we initialize the presence list with 6 bits set to 0, which means that the user is considered absent during all time slots

    for cell, stamp in zip(cells, stamps):
        for i, (start, end) in enumerate(window.values()):
            if start <= stamp < end: # if the timestamp is within the time slot, we set the corresponding bit to 1, which means that the user is considered present during this time slot
                presence[i] = 1
    
    presence_code = 0
    for i, bit in enumerate(presence):
        presence_code += bit * 2 **i # we convert the binary code to a decimal number, which will be easier to manipulate and store in the dataframe

    return presence_code

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

            presence_code = classify(user_cells, user_stamps)

            result_day["user_id"].append(user_id)
            result_day["day"].append(day)
            result_day["user_presence_classification"].append(presence_code)

            

df = pd.DataFrame(result_day)

df = df.groupby(by=["day","user_presence_classification"]).size().unstack(fill_value=0)

df.to_csv(OUTPUT_DIR / "user_generalised_classification.csv", sep=";", header=True, index=True, index_label=True)



