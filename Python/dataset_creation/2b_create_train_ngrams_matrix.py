# V2
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
from pathlib import Path
from collections import Counter, defaultdict

import utils

MAIN_DIR = Path(__file__).parent.parent.parent
TRAIN_TEST_PATH = MAIN_DIR / "results/predictions/train_test"
OUTPUT_DIR = MAIN_DIR / f"results/predictions/simple_predictor/transition_matrix"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_DELTA = 14400 + 30
CONTEXT_LENGTH = [1,2,3,4,5]

def fill_dict(res, lim_res, data):
    for line in tqdm.tqdm(data):
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
                    
                    res[gram][seq_without_last].update([last])

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
        output_path = BASE_DIR / (inner_dir)
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

def gen_training_matrix(train_set_path=None, val_set_path=None, output_path=None, inner_dir="", filename=None):
    # Transform data
    processed_train_data = {gram : defaultdict(Counter) for gram in CONTEXT_LENGTH}
    limited_processed_train_data = {gram : defaultdict(Counter) for gram in CONTEXT_LENGTH}
    train_data = open_csv(path = train_set_path)
    if val_set_path:
        val_data   = open_csv(path = val_set_path)
        train_data += val_data
        val_data = None
    
    fill_dict(processed_train_data,limited_processed_train_data, train_data)

    # Sort every suffix dict by number of occurence in descending order for convenience
    for gram in CONTEXT_LENGTH:
        for k,v in processed_train_data[gram].items():
            processed_train_data[gram][k] = dict(sorted(v.items(),key = lambda x : x[1], reverse=True))
        for k,v in limited_processed_train_data[gram].items():
            limited_processed_train_data[gram][k] = dict(sorted(v.items(),key = lambda x : x[1], reverse=True))

    save_dict(processed_train_data, output_path, inner_dir=inner_dir, file_name=filename)
    save_dict(limited_processed_train_data, output_path, inner_dir=inner_dir, file_name="limited_" + filename)

if __name__ == "__main__":
    
    # # Random 80/20
    gen_training_matrix(train_set_path  = TRAIN_TEST_PATH / "train_random.csv",
                        output_path = OUTPUT_DIR, inner_dir = "training_matrix", filename = "ngrams_matrix_train_random.json")
    
    # Random 80/20 - removed repeat
    # gen_training_matrix(train_set_path  = TRAIN_TEST_PATH / "removed_repeat/train_random_removed_repeat.csv",
    #               output_path = OUTPUT_DIR, inner_dir = "training_matrix/removed_repeat", filename = "ngrams_matrix_train_random_removed_repeat.json")
    
    
    # Random 80/20 2g3g merge
    # gen_training_matrix(train_set_path  = TRAIN_TEST_PATH / "merge/train_random.csv",
    #                      output_path = OUTPUT_DIR, inner_dir = "training_matrix/merge", filename = "2g3g_matrix_train_random.json")
    
    # Random 80/20 2g3g merge - removed repeat
    # gen_training_matrix(train_set_path  = TRAIN_TEST_PATH / "merge/train_random_removed_repeat.csv",
    #                       output_path = OUTPUT_DIR, inner_dir = "training_matrix/merge", filename = "2g3g_matrix_train_random_removed_repeat.json")
    
    
    
    # # Age 60/40
    # gen_training_matrix(train_set_path  = TRAIN_TEST_PATH / "train_age.csv",
    #                     output_path = OUTPUT_DIR, inner_dir = "training_matrix", filename = "ngrams_matrix_train_age.json")
    
    
    # # Class 1
    # gen_training_matrix(train_set_path  = TRAIN_TEST_PATH / "class/class1_train.csv",
    #                     output_path = OUTPUT_DIR, inner_dir = "training_matrix/class", filename = "matrix_train_class1.json")
    
    # # Weekdays
    # gen_training_matrix(train_set_path  = TRAIN_TEST_PATH / "week/weekdays_train.csv",
    #                     output_path = OUTPUT_DIR, inner_dir = "training_matrix/week", filename = "matrix_train_weekdays.json")
    
    # # Weekend
    # gen_training_matrix(train_set_path  = TRAIN_TEST_PATH / "week/weekend_train.csv",
    #                     output_path = OUTPUT_DIR, inner_dir = "training_matrix/week", filename = "matrix_train_weekend.json")
    
    # # Class 1 + weekdays
    # gen_training_matrix(train_set_path  = TRAIN_TEST_PATH / "class_week/class1_weekdays_train.csv",
    #                     output_path = OUTPUT_DIR, inner_dir = "training_matrix/class_week", filename = "matrix_train_class1_weekdays.json")
    
    # # Class 1 + weekend
    # gen_training_matrix(train_set_path  = TRAIN_TEST_PATH / "class_week/class1_weekend_train.csv",
    #                     output_path = OUTPUT_DIR, inner_dir = "training_matrix/class_week", filename = "matrix_train_class1_weekend.json")