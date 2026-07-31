# V1
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

import tqdm
import csv
import json
import time
from pathlib import Path
from collections import Counter, defaultdict

import utils

MAIN_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = MAIN_DIR / "Database/no_duplicate"

OUTPUT_DIR = MAIN_DIR / f"results/simple_predictor/transition_matrix"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_DELTA = 14400 + 30
files = [path for path in DATASET_DIR.glob("*.csv")]

def fill_dict(res, lim_res, data):

    for line in tqdm.tqdm(data, desc=f"Computing 2,3,4,5,6-grams for {file.name}"):
        stations = [c for c in line[8::2]]
        timestamps = [int(ts) for ts in line[9::2]]
        for gram in CONTEXT_LENGTH:
            
            # ============
            # All res
            # ============
            ngrams = ['-'.join(seq) for seq in set(utils.find_ngrams(stations,gram+1))] # Unique ngrams

            if len(ngrams) != 0:
                for seq in ngrams:
                    last = seq.split("-")[-1]
                    seq_without_last = seq[:len(seq)-len(last)-1]
                    
                    # The next line counts the number of time the truncated sequence "seq_without_last" --which is "seq" 
                    # (one of the n-gram found in "stations") without the last cell of the sequence-- ends with "last".
                    # This allows to give us the probability of the cell "last" to appear after seeing the sequence "seq_without_last"
                    res[gram][seq_without_last].update([last]) # +1 for n-gram without the last cell that must be predicted 

                # =============
                # Limited res
                # =============
                ngrams2 = utils.find_ngrams_optimized(stations,gram+1,timestamps=timestamps,max_gap=MAX_DELTA) # Must respect MAX_DELTA between consecutive timestamps
                if len(ngrams2) !=0:                     
                    ngrams2 = utils.transform_ngrams_list_to_seq_list(ngrams2)
                    for seq in ngrams2:
                        last = seq.split("-")[-1]
                        seq_without_last = seq[:len(seq)-len(last)-1]
                        lim_res[gram][seq_without_last].update([last])
    
def save_dict(data, BASE_DIR : Path, inner_dir : str | None, file_name : str):
    output_path = BASE_DIR
    if inner_dir is not None:
        output_path = BASE_DIR / inner_dir
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / file_name
    with open(output_path, mode="w", encoding='utf-8') as f:
        json.dump(data,f,indent=2)

# CONTEXT_LENGTH is 1 to 5 and ngrams are 2 to 6 with the last element being the one to be predicted
CONTEXT_LENGTH = [1,2,3,4,5]
processed_train_data = {gram : defaultdict(Counter) for gram in CONTEXT_LENGTH}
limited_processed_train_data = {gram : defaultdict(Counter) for gram in CONTEXT_LENGTH}

for file in files:
    day = utils.get_day(file)
    with open(file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=";")
        csv_data = list(reader)
    fill_dict(processed_train_data,limited_processed_train_data, data=csv_data)

for gram in CONTEXT_LENGTH:
    for k,v in processed_train_data[gram].items():
        processed_train_data[gram][k] = dict(sorted(v.items(),key = lambda x : x[1], reverse=True))
    for k,v in limited_processed_train_data[gram].items():
        limited_processed_train_data[gram][k] = dict(sorted(v.items(),key = lambda x : x[1], reverse=True))

print("Saving dicts...")
t0 = time.time()
save_dict(processed_train_data, OUTPUT_DIR, inner_dir="ngrams", file_name="transition_ngrams_number.json")
save_dict(limited_processed_train_data, OUTPUT_DIR, inner_dir="ngrams", file_name="limited_transition_ngrams_number.json")
print(f"Saved in {time.time()-t0:.2f}s")
