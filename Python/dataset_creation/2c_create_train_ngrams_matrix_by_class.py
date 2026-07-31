# V2
# Same as transition_matrix but for longer context (deeper transition matrices) 
# for n = 2,3,4,5,6, it outputs dictionnaries as json format with the following shape:
# { 
#   LONGUEUR_CONTEXTE_i : 
#       {
    #   CLASSE : {
    #      SEQUENCE_CELLULE_CONTEXTE : 
    #               {
    #           CELLULE_SUFFIXE : NB DE FOIS QUE CE SCHEMA A ETE TROUVE
    #               }
    #           }
#       }
# }
#

import tqdm
import csv
import json
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

import utils

MAIN_DIR = Path(__file__).parent.parent.parent
TRAIN_TEST_PATH = MAIN_DIR / "results/predictions/train_test"
CLASSIFICATION_PATH = MAIN_DIR / "results/intermediate_result/cell_classification.csv"
OUTPUT_DIR = MAIN_DIR / f"results/predictions/simple_predictor/transition_matrix"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_DELTA = 14400 + 30
CONTEXT_LENGTH = [1,2,3,4,5]

def fill_dict(res, lim_res, user_classification, data):
    for line in tqdm.tqdm(data):
        stations = [c for c in line[8::2]]
        timestamps = [int(ts) for ts in line[9::2]]
        for gram in CONTEXT_LENGTH:

            # ============
            # All res
            # ============
            ngrams = ['-'.join(seq) for seq in set(utils.find_ngrams(stations,gram+1))] # Unique ngrams

            if len(ngrams) != 0:
                user_class = user_classification.loc[int(line[0])].user_presence_classification
                for seq in ngrams:
                    last = seq.split("-")[-1]
                    seq_without_last = seq[:len(seq)-len(last)-1]
                    
                    res[gram][int(user_class)][seq_without_last].update([last])

                # =============
                # Limited res
                # =============
                ngrams2 = utils.find_ngrams_optimized(stations,gram+1,timestamps=timestamps,max_gap=MAX_DELTA) # Must respect MAX_DELTA between consecutive timestamps
                if len(ngrams2) !=0:                     
                    ngrams2 = utils.transform_ngrams_list_to_seq_list(ngrams2)
                    for seq in ngrams2:
                        last = seq.split("-")[-1]
                        seq_without_last = seq[:len(seq)-len(last)-1]
                        lim_res[gram][int(user_class)][seq_without_last].update([last])
    
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
processed_train_data = {gram : defaultdict(lambda : defaultdict(Counter)) for gram in CONTEXT_LENGTH}
limited_processed_train_data = {gram : defaultdict(lambda : defaultdict(Counter)) for gram in CONTEXT_LENGTH}
train_data = open_csv(path = TRAIN_TEST_PATH / "train_random.csv")
user_classification = pd.read_csv(CLASSIFICATION_PATH,sep=";", index_col=["user_id"])
fill_dict(processed_train_data,limited_processed_train_data, user_classification, train_data)

# Sort every suffix dict by number of occurence in descending order for convenience
for gram in CONTEXT_LENGTH:
    for classes in processed_train_data[gram].keys():
        for k,v in processed_train_data[gram][classes].items():
            processed_train_data[gram][classes][k] = dict(sorted(v.items(),key = lambda x : x[1], reverse=True))
        for k,v in limited_processed_train_data[gram][classes].items():
            limited_processed_train_data[gram][classes][k] = dict(sorted(v.items(),key = lambda x : x[1], reverse=True))

save_dict(processed_train_data, OUTPUT_DIR, inner_dir="ngrams_by_class", file_name="transition_ngrams_train_class.json")
save_dict(limited_processed_train_data, OUTPUT_DIR, inner_dir="ngrams_by_class", file_name="limited_transition_ngrams_train_class.json")