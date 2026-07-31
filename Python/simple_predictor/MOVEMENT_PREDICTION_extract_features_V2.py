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
from models import MovementPredictor
from collections import Counter

MAIN_DIR = Path(__file__).parent.parent.parent

def extract_features_from_path(path : Path):
    extracter = MovementPredictor()
    
    dataframe =\
    {
        "label" : [],

            # Moyenne de temps entre chaque record --> Si elle est haute, surement des enregistrements systèmes (pas de mouvement)
        "mean_time_between_record" : [],

        # Nb de record par heure (peut être très fort en le mettant en lien avec le temps des derniers timestamp)
        # --> Si 5 timestamp seulement à 16h --> Quelqu'un de surement inactif --> Ne bouge pas
        # Par contre ça nous dit rien s'il a 50 timestamp à 16h, ça nous aide à trouver les cas où les gens ne bougent jamais
        # mais pas les mouvements des gens
        "record_per_hour" : [],

        # Capture si l'utilisateur a réellement une attache à sa première cellule de la journée
        "number_of_time_in_first_cell" : [],


        # Moving - est-ce que la personne bouge en ce moment ? (on regarde les 2 dernières transitions, doit pas être la même station)
        "moving" : [],
        
        # -- Position vs ancre --
        "is_at_anchor_cell": [],
        "trip_phase": [],
                    
        # -- Déclenchement du mouvement --
        "has_departed_today":   [],
        "first_departure_elapsed_h": [], 
        "time_in_anchor_before_dep_h":  [],
        
        # -- Retour à l'ancre --
        "returned_to_anchor": [],   
        "time_since_return_h":  [],
        
        # -- Dynamique du mouvement --
        
        "local_acceleration": [],         
        "n_complete_trips": [], 
        "current_trip_duration_h": [],
        
        # -- Profil de mobilité --
        "out_of_anchor_entropy": []
    }

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
                    
                movement = int(history_records[-1] != true_next)
                dataframe['label'].append(movement)
                    
                features = extracter.extract_features(history_records, history_timestamps)
                    
                    
                time_diff1 = timestamps[random_split-1]
                    
                dataframe["mean_time_between_record"].append(np.mean(np.diff(history_timestamps)))                    
                dataframe["record_per_hour"].append(random_split / time_diff1) # record/h moyen par rapport au dernier record avant la prédiction
                dataframe["number_of_time_in_first_cell"].append(Counter(cells[:random_split])[cells[0]])
                moving = cells[random_split-1] != cells[random_split-2] and cells[random_split-2] != cells[random_split-3]
                dataframe["moving"].append(moving)

                for k,v in features.items():
                    dataframe[k].append(v)

    df = pd.DataFrame(dataframe)
    df = df.astype({
        "label" : pd.Int16Dtype(), 

        "mean_time_between_record" : pd.Float32Dtype(),
        "record_per_hour" : pd.Float32Dtype(),
        "number_of_time_in_first_cell" : pd.Int16Dtype(),
        "moving" : pd.Int16Dtype(),


        # -- Position vs ancre --
        "is_at_anchor_cell": pd.Int16Dtype(),
        "trip_phase": pd.Int16Dtype(),                  
        # -- Déclenchement du mouvement --
        "has_departed_today": pd.Int16Dtype(),   
        "first_departure_elapsed_h": pd.Float32Dtype(), 
        "time_in_anchor_before_dep_h": pd.Float32Dtype(),
        # -- Retour à l'ancre --
        "returned_to_anchor": pd.Int16Dtype(),   
        "time_since_return_h": pd.Float32Dtype(),
        # -- Dynamique du mouvement --
        "local_acceleration": pd.Float32Dtype(),         
        "n_complete_trips": pd.Int16Dtype(), 
        "current_trip_duration_h": pd.Float32Dtype(),
        # -- Profil de mobilité --
        "out_of_anchor_entropy": pd.Float32Dtype()
    })
    df.to_csv(OUTPUT_DIR / f"{path.stem}_features_V6.csv", sep=";", header=True, index=False)

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

OUTPUT_DIR = MAIN_DIR / "results/predictions/movement_prediction/features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CONTEXT = 5
MAX_CONTEXT = 512

train_csv = MAIN_DIR / "results/predictions/train_test/class_merge/class1_train_random.csv"
extract_features_from_path(path=train_csv)
# extract_only_late_features_from_path(path=train_csv)

test_csv = MAIN_DIR / "results/predictions/train_test/class_merge/class1_test_random.csv"
extract_features_from_path(path=test_csv)
# extract_only_late_features_from_path(path=test_csv)










