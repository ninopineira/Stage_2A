from pathlib import Path
import json
import csv
import math
import time
import pandas as pd
from geopy.distance import geodesic
from collections import Counter
import datetime

def merge_cell_id(cell : str):
    if cell.startswith(('B','D')):
        return cell[1:-1]
    else:
        return cell[1:-3]

def convert_lat_lon_distance_to_meter(lat1 : float, lat2 : float, lon1 : float, lon2 : float) -> float :
    """
    Returns the distance between 2 points at the surface of the Earth in meters
    """    
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)

    distance = geodesic(point1, point2).meters
    return distance

def convert_lat_lon_distance_to_meter(point1 : tuple[float], point2 : tuple[float]) -> float :
    """
    Returns the distance between 2 points at the surface of the Earth in meters
    """    
    distance = geodesic(point1, point2).meters
    return distance

def get_lat_lon_cell(dataframe : pd.DataFrame, cellid : str) :
    return dataframe[dataframe["cellid"] == cellid]["lat"].values[0], dataframe[dataframe["cellid"] == cellid]["lon"].values[0]


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


class HelperVOMM:
    """
    Here to convert pre-computed files and other stuffs into the right format for the VOMM class
    """
    def __init__(self):
        pass
    
    def convert_prefix_string_to_tuple(self,sequence : str) -> tuple[str]:
        """
        Converts prefix strings from "A-B-C" to ("A","B","C")
        """
        return tuple(sequence.split("-"))
    
    def prepare_ngrams(self,train_data : dict) -> dict:
        
        dict_context_totals = {}
        dict_unique_followers = {}
            
        for k in train_data.keys(): # values of n in "n-grams"
            gram = k
            dict_context_totals[gram] = {}
            dict_unique_followers[gram] = {}
            for seq,v in train_data[k].items():
                dict_context_totals[gram][seq] = sum(v.values())
                dict_unique_followers[gram][seq] = len(v)
                
        return dict_context_totals, dict_unique_followers

    def prepare_unigram_counts(self,dataset_filepath):
        unigram_counts = Counter()
        files =  [path for path in dataset_filepath.glob("*.csv")]
        for file in files:
            with open(file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f, delimiter=";")
                for line in reader:
                    #stations = {merge_cell_id(c) for c in line[8::2]}
                    stations = {c for c in line[8::2]}
                    unigram_counts.update(stations)
        return unigram_counts

class HelperData:
    """
    Utility functions for the original dataset
    """
    def __init__(self):
        pass
    
    def map_users_to_day_and_file_row(self,dataset_filepath : Path, output_path : Path) -> dict:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        files =  list(dataset_filepath.glob("*.csv"))
        res_dict = {}
        for file in files:
            day = self.get_day(filepath=file)
            with open(file, mode='r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f, delimiter=";")
                for row in reader:
                    res_dict[row[0]] = [day,reader.line_num]

        with open(output_path, mode="w") as f:
            json.dump(res_dict,f,indent=2)
        print(f"Saved the mapping user_id -> (day,row_number) at : {output_path}")    
    
    def get_day(self,filepath):
        return filepath.name.split("_")[0]
    
    def save_dict(self, data, BASE_DIR : Path, inner_dir : str | None, file_name : str):
        output_path = BASE_DIR
        if inner_dir is not None:
            output_path = BASE_DIR / inner_dir
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / file_name
        with open(output_path, mode="w", encoding='utf-8') as f:
            json.dump(data,f,indent=2)

    def save_csv(self, data, path):
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerows(data)

class Metrics:
    def __init__(self):
        pass
    
    # 2 Versions of top_k_acc V1 if only need top_k in main script, save some space, V2 if you need preds_list
    def top_k_accuracy(self, top_k : list, true_next : list, k=1):
        """
        top_k :  [ [(pred_cell,prob), ...], ... ] 
        true_next : [ true_cell,... ]
        """
        
        correct = 0
        total = len(top_k)
        
        assert len(top_k) == len(true_next), "top_k and true_next must have the same length"
        
        print(f"Computing ACC@{k} for {len(top_k)} predictions")
        t0 = time.time()
        for top,t_next in zip(top_k, true_next):
            for i in range(min(k,len(top))):
                if top[i][0] == t_next:
                    correct += 1
                    break
        print(f"Computed ACC@{k} in {time.time()-t0:.2f}s")
        
        return correct / total if total > 0 else 0
        
    def top_k_accuracy(self, preds_list : list, true_next : list, k=1):
        """
        preds_list : [ [(pred_cell,prob), ...], ... ] -> Liste complète non découpée
        true_next  : [ true_cell,... ]
        """
        correct = 0
        total = len(preds_list)
        
        assert len(preds_list) == len(true_next), "preds_list and true_next must have the same length"
        
        print(f"Computing ACC@{k} for {total} predictions")
        t0 = time.time()
        
        # On utilise zip sur les listes brutes
        for preds, t_next in zip(preds_list, true_next):
            for i in range(min(k, len(preds))):
                if preds[i][0] == t_next:
                    correct += 1
                    break
                    
        print(f"Computed ACC@{k} in {time.time()-t0:.2f}s")
        
        return correct / total if total > 0 else 0
     
     
    # 2 Versions of map_k V1 if only need top_k in main script, save some space, V2 if you need preds_list
    def map_k(self, top_k : list, true_next : list, k=1):
        total = len(top_k)
        score = 0.0

        assert total == len(true_next)

        print(f"Computing MAP@{k} for {len(top_k)} predictions")
        t0 = time.time()
        for top, t_next in zip(top_k, true_next):
            for i in range(min(k,len(top))):
                if top[i][0] == t_next:
                    score += 1.0 / (i + 1)
                    break
        print(f"Computed MAP@{k} in {time.time()-t0:.2f}s")

        return score / total if total > 0 else 0
    
    def map_k(self, preds_list : list, true_next : list, k=1):
        total = len(preds_list)
        score = 0.0

        assert total == len(true_next)

        print(f"Computing MAP@{k} for {len(preds_list)} predictions")
        t0 = time.time()
        for top, t_next in zip(preds_list, true_next):
            for i in range(min(k,len(top))):
                if top[i][0] == t_next:
                    score += 1.0 / (i + 1)
                    break
        print(f"Computed MAP@{k} in {time.time()-t0:.2f}s")

        return score / total if total > 0 else 0





    # Evaluate the quality of the prediction through the probability value given by the model
    # If the model is really confident and make a mistakes, the error will be big.
    def negative_log_likelihood(self, model, sequences, epsilon=1e-12):
        total_loss = 0
        total = 0
        
        for seq in sequences:
            for i in range(1, len(seq)):
                context = seq[max(0, i-model.max_order):i]
                true_next = seq[i]
                
                p = model.prob(context, true_next)
                p = max(p, epsilon)
                
                total_loss += -math.log(p)
                total += 1
    
        return total_loss / total
    
    def log_likelihood(self, preds_list, true_next_list, epsilon=1e-15):
        total_loss = 0
        total = len(preds_list)
        worst_score = -math.log(epsilon, 10)
        for k in range(total):
            d = dict(preds_list[k])
            true_next = true_next_list[k]
            if true_next in d:
                p = d[true_next]
                total_loss += -math.log(p, 10)
            else :
                total_loss += worst_score # = -15.0 
        
        return total_loss / total



    def perplexity(self, model, sequences):
        ll = self.negative_log_likelihood(model, sequences)
        return math.exp(ll)

    def mean_reciprocal_rank(self,model, sequences):
        total_rr = 0
        total = 0
        
        for seq in sequences:
            for i in range(1, len(seq)):
                context = seq[:i]
                true_next = seq[i]
                
                preds, _ = model.predict_next(context, top_k=None)
                
                for rank, (state, _) in enumerate(preds, start=1):
                    if state == true_next:
                        total_rr += 1 / rank
                        break
                
                total += 1
        
        return total_rr / total if total > 0 else 0



# ================================================================#
# ================================================================#
#                           DATASETS                              #
# ================================================================#
# ================================================================#

# Simple dataset for prediction

from torch.utils.data import Dataset
import torch
class MobilityDataset(Dataset):

    def __init__(self, samples_path):

        t0 = time.time()
        print(f"Loading samples ... {samples_path}")
        self.samples = torch.load(samples_path)
        print(f"loaded samples in {time.time()-t0:.2f}s")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        return row[0], row[1], row[2]

