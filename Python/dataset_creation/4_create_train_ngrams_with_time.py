# V3
# Same as transition_matrix but for longer context (deeper transition matrices) 
# for n = 2,3,4,5,6, it outputs dictionnaries as json format with the following shape:
# { 
#   LONGUEUR_CONTEXTE_i : 
#   {
#      SEQUENCE_CELLULE_CONTEXTE : 
#       {
#           CELLULE_SUFFIXE : NB DE FOIS QUE CE SCHEMA A ETE TROUVE
#       }
#   }
# }
#
# transition time matrix : { longueur contexte : { contexte : {cellule_suffixe : délai} }  }

# /!\ TAKES AROUND 1 HOUR OF COMPUTATION AND DUMPING

import tqdm
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

import utils

MAIN_DIR = Path(__file__).parent.parent.parent
TRAIN_TEST_PATH = MAIN_DIR / "results/predictions/train_test"
OUTPUT_DIR = MAIN_DIR / f"results/predictions/simple_predictor/transition_matrix"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_DELTA = 14400 + 30
CONTEXT_LENGTH = [1,2,3,4,5]

def fill_dict(time, data):
    for line in tqdm.tqdm(data):
        stations = [c for c in line[8::2]]
        timestamps = [int(ts) for ts in line[9::2]]
        for gram in CONTEXT_LENGTH:
            # Unique ngrams with their timestamps
            ngrams_with_ts = utils.find_ngrams_with_timestamps_unique_fast(stations, timestamps, gram + 1)
            if len(ngrams_with_ts) != 0:
                for seq,ts in ngrams_with_ts:
                    last = seq.split("-")[-1]
                    seq_without_last = seq[:len(seq)-len(last)-1]
                                        
                    # Couple (temps moyen)
                    diffs = np.diff(ts[:-1])
                    if len(diffs) == 0:
                        time[gram][seq_without_last][last].append( (0,  ts[-1] - ts[-2]) )
                    else:
                        last_ctx_gap = int(ts[-2] - ts[-3]) if len(ts) >= 3 else 0
                        time[gram][seq_without_last][last].append((
                            float(np.mean(diffs)) if len(diffs) > 0 else 0.0,  # mean
                            int(ts[-1] - ts[-2]),                                # suffix gap
                            last_ctx_gap,                                        # dernier gap du contexte
                        ))
    
def save_dict(data, BASE_DIR : Path, inner_dir : str | None, file_name : str):
    output_path = BASE_DIR
    if inner_dir is not None:
        output_path = BASE_DIR / inner_dir
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / file_name
    with open(output_path, mode="w", encoding='utf-8') as f:
        json.dump(data,f,indent=2)

def open_csv(path):
    data = None
    with open(path, mode = 'r', newline='') as f:
        reader = csv.reader(f, delimiter=";")
        data = list(reader)
    assert data is not None, "data should not be None"
    return data

# Transform data
processed_train_data_time = {gram : defaultdict(lambda : defaultdict(list)) for gram in CONTEXT_LENGTH}
train_data = open_csv(path = TRAIN_TEST_PATH / "train_random.csv")
fill_dict(processed_train_data_time, train_data)
train_data = None

with open("time.json", "w") as f:
    json.dump(processed_train_data_time, f, indent=2)