# Extract the features from the train csv specified to produce a training dataset for a machine learning model to learn
# to classify movement or not depending on these features.

# The only realistic way to do that is to first merge the cells a base station, like said by the tutor, we need
# to find the time where they start moving and the time when they stop.


import tqdm
import csv
import pandas as pd
import random
from pathlib import Path
import numpy as np
from models import MovementPredictor, MovementPredictorV2
from collections import Counter

MAIN_DIR = Path(__file__).parent.parent.parent

def extract_features_from_path(path : Path):
    """Extract the MovementPredictorV2 feature set (incl. the causal entropy
    features) at one random prediction point per user, plus the move/stay label.
    Everything is computed on history_records = cells[:random_split] only, so the
    features are strictly causal (no leakage from the future)."""
    extracter = MovementPredictorV2()
    rows = []

    with open(path, mode="r", newline='') as f:
        reader = csv.reader(f, delimiter=";")
        for user in tqdm.tqdm(reader, desc="Extracting features from set..."):
            n_preds = int(user[7])
            if MAX_CONTEXT > n_preds > MIN_CONTEXT:
                cells = user[8::2]
                timestamps = [int(ts) for ts in user[9::2]]

                random_split = random.randint(MIN_CONTEXT, n_preds - 1)
                history_records = cells[:random_split]
                history_timestamps = timestamps[:random_split]

                true_next = cells[random_split]
                movement = int(history_records[-1] != true_next)   # 1 = moves next

                features = extracter.extract_features(history_records, history_timestamps)
                rows.append({"label": movement, **features})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / f"{path.stem}_features.csv", sep=";", header=True, index=False)

def extract_only_late_features_from_path(path : Path):
    extracter = MovementPredictor()
    
    dataframe =\
    {"label" :                       [],
        # -- Local stationarity --
        "current_cell_duration_h":      [], 
        "current_streak":               [],          
        "recent_variability":           [],      
        "daily_transition_rate":        [],   
        "time_since_last_trans_h":      [], 
        "inter_transition_rhythm_h":    [],
        
        # -- User profile --
        "concentration":                [], 
        "momentum":                     [], 
        "is_home_now":                  [], 
        "is_activity_now":              [], 
        "home_hour_match":              [], 
        "activity_hour_match":          [], 
        "n_distinct_today":             []}

    with open(path, mode="r", newline='') as f:
        reader = csv.reader(f, delimiter=";")
        for user in tqdm.tqdm(reader, desc="Extracting features from set..."):
            n_preds = int(user[7])
            if n_preds > 5:
                cells = user[8::2]
                timestamps = [int(ts) for ts in user[9::2]]
                try:
                    late_timestamp_index = next(x[0] for x in enumerate(timestamps) if x[1] > 57600)
                    if late_timestamp_index <= n_preds - 2 :
                        home_cell = user[5] if user[5] != '' else None
                        activity_cell = user[6] if user[6] != '' else None
                        
                        for histo in range(late_timestamp_index, n_preds - 1):
                            history_records = cells[:histo]
                            history_timestamps = timestamps[:histo]
                            true_next = cells[histo]
                            
                            movement = int(history_records[-1] != true_next)
                            
                            dataframe['label'].append(movement)
                            
                            features = extracter.extract_features(history_records, history_timestamps, home_cell, activity_cell)
                            for k,v in features.items():
                                dataframe[k].append(v)
                except:
                    continue

    df = pd.DataFrame(dataframe)
    df = df.astype({
        "label" : pd.Int16Dtype(), 
        # -- Local stationarity --
        "current_cell_duration_h": pd.Float32Dtype(),
        "current_streak": pd.Int16Dtype(),
        "recent_variability": pd.Float32Dtype(),
        "daily_transition_rate": pd.Float32Dtype(),
        "time_since_last_trans_h": pd.Float32Dtype(),
        "inter_transition_rhythm_h": pd.Float32Dtype(),
            

        # -- User profile --
        "concentration":  pd.Float32Dtype(),
        "momentum":   pd.Float32Dtype(),
        "is_home_now": pd.Int16Dtype(),
        "is_activity_now": pd.Int16Dtype(),       
        "home_hour_match": pd.Float32Dtype(),       
        "activity_hour_match": pd.Float32Dtype(),   
        "n_distinct_today": pd.Int16Dtype(), 
    })
    df.to_csv(OUTPUT_DIR / f"{path.stem}_features_extralate.csv", sep=";", header=True, index=False)

OUTPUT_DIR = MAIN_DIR / "results/predictions/movement_prediction_V2/features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CONTEXT = 5
MAX_CONTEXT = 512

train_csv = MAIN_DIR / "results/predictions/train_test/class_merge/class1_train_random.csv"
extract_features_from_path(path=train_csv)
# extract_only_late_features_from_path(path=train_csv)

test_csv = MAIN_DIR / "results/predictions/train_test/class_merge/class1_test_random.csv"
extract_features_from_path(path=test_csv)
# extract_only_late_features_from_path(path=test_csv)










