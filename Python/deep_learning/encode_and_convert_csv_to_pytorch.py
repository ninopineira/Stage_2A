# Encode cellids and timestamps for the MobilityDataset class of the deep_learning folder for Transformer based models

import tqdm
import csv
import json
from pathlib import Path
import datetime
import torch

torch.manual_seed(67)

MAIN_DIR = Path(__file__).parent.parent.parent
dataset_name = Path(__file__).parent.name

# Data to store
INPUT_PATH = MAIN_DIR / f"Database/no_duplicate_max_512_records"
MAP_PATH = MAIN_DIR / "results/predictions/deep_learning/cell_map_start_1.json"

OUTPUT_PATH_DIR = MAIN_DIR / f"results/predictions/deep_learning"
OUTPUT_PATH_DIR.mkdir(parents=True, exist_ok=True)

FIXED_LEN = 512   # Maximum number of records for the users each day 99.4% of the users have less than 512 records, it's 2^9 for better computation
MIN_LEN = 8
DAY_SECOND = 86400.0
PAD_CELL_ID = 0
PAD_TIMESTAMPS = 0.0
EOS_OFFSET = 1

TEST_PART = 0.2
VAL_PART  = 0.4

# ---------------------------------------------------------------------------
# Préparation d'une séquence utilisateur → tenseurs
# ---------------------------------------------------------------------------

def str_to_int_cells(cell_map, cell_str):
    return [cell_map[cellid] for cellid in cell_str]
    
def encode_sequence(
    cell_ids: list[int],        # déjà encodés en entiers 1..N
    timestamps: list[int],      # secondes 0-86399
    eos_id: int,
    max_len: int = FIXED_LEN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ajoute EOS, tronque, puis padde à max_len.
 
    Retourne
    --------
    cells_t   : LongTensor (max_len,)  — 0 = pad, 1..N = cellule, N+1 = EOS
    times_t   : FloatTensor (max_len,) — secondes normalisées [0, 1]
    mask_t    : BoolTensor (max_len,)  — True sur positions paddées
    """
    # Ajout EOS
    cells = list(cell_ids) + [eos_id]
    times = list(timestamps) + [0.0]
 
    # Troncature
    cells = cells[:max_len]
    times = times[:max_len]
 
    seq_len = len(cells)
    pad_len = max_len - seq_len
 
    # Padding à droite
    cells += [PAD_CELL_ID]      * pad_len
    times += [PAD_TIMESTAMPS]   * pad_len
 
    cells_t = torch.tensor(cells, dtype=torch.uint16) # STORE AS SHORT BUT LONG ARE NEEDED FOR CRITERION
    times_t = torch.tensor(times, dtype=torch.float32).div_(DAY_SECOND)   # normalise [0,1]
    mask_t  = torch.zeros(max_len, dtype=torch.bool)
    if pad_len > 0:
        mask_t[seq_len:] = True
 
    return cells_t, times_t, mask_t

def get_day(filepath : Path) -> str:
    return filepath.name.split("_")[0]

def is_weekend(day : str) -> bool:
    """
    Determine if day is part of a weekend or not.
    
    ONLY FORMAT ACCEPTED FOR INPUT IS : "YYYY-MM-DD"
    """
    split_day = day.split("-")
    day_object = datetime.date(year = int(split_day[0]),
                  month = int(split_day[1]),
                  day = int(split_day[2]))
    return day_object.weekday() > 4

with open(MAP_PATH) as f:
    cell_map = json.load(f) 


n_cells = max(cell_map.values())
files = list(INPUT_PATH.glob("*.csv"))
cells_list = []
times_list = []
mask_list = []

for file in files:
    day = get_day(file)
    weekend = is_weekend(day)
    with open(file,mode="r") as f:
        reader = csv.reader(f,delimiter=";")
        for line in tqdm.tqdm(list(reader), desc=f"Encoding for all users of day {day}", colour="blue"):
            user_id = line[0]
            n_record = int(line[7])
            if n_record < MIN_LEN:
                continue
            cells = line[8::2]
            cells = str_to_int_cells(cell_map=cell_map, cell_str=cells)
            timestamps = [int(ts) for ts in line[9::2]]
            
            cells_t, times_t, mask_t = encode_sequence(cell_ids = cells, timestamps=timestamps, eos_id=n_cells+EOS_OFFSET, max_len=FIXED_LEN)

            cells_list.append(cells_t)
            times_list.append(times_t)
            mask_list.append(mask_t)

n_users = len(cells_list)
print("Nb users", n_users)
random_indices = torch.randperm(n_users)

cells_all = torch.stack(cells_list)[random_indices]   # [N, L]
times_all = torch.stack(times_list)[random_indices]   # [N, L]
mask_all  = torch.stack(mask_list)[random_indices]    # [N, L]

test_split, val_split = int(n_users*TEST_PART), int(n_users*VAL_PART)

OUTPUT_TRAIN = OUTPUT_PATH_DIR / "encoded_train.pt"
OUTPUT_TEST = OUTPUT_PATH_DIR / "encoded_test.pt"
OUTPUT_VAL = OUTPUT_PATH_DIR / "encoded_val.pt"

# Save train
torch.save({"cells": cells_all[val_split:,:].clone(),
        "times": times_all[val_split:,:].clone(),
        "mask": mask_all[val_split:,:].clone()},OUTPUT_TRAIN)

# Save test
torch.save({"cells": cells_all[:test_split,:].clone(),
        "times": times_all[:test_split,:].clone(),
        "mask": mask_all[:test_split,:].clone()},OUTPUT_TEST)

# Save val
torch.save({"cells": cells_all[test_split:val_split,:].clone(),
        "times": times_all[test_split:val_split,:].clone(),
        "mask": mask_all[test_split:val_split,:].clone()},OUTPUT_VAL)

print(f"Tout a été enregistré avec succès (bravo)")