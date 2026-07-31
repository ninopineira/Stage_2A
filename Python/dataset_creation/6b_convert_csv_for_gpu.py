import csv
import json
import tqdm
from pathlib import Path

import numpy as np
import torch

# Convertir tout en float pytorch parce que gpu aime que ce qui flotte
# On va faire 1 dataset par order tel que du coup on aura 1 ligne = 1 prédiction à faire et comme ça dépend de 
# la taille du contexte il faut faire 1 fichier par longueur de contexte
def main(users, cell_to_idx, ctxt_to_idx, order, output_sample_path):
    
    samples = []

    contexts = []
    hours = []
    targets = []

    for user in tqdm.tqdm(users, desc=f"Converting user data into samples...", colour="green"):
        n_cells = int(user[7])
        if n_cells < 6 : # Skip les gens avec peu de records car chiant
            continue
        cells = user[8::2]
        timestamps = np.array(user[9::2], dtype=np.int32) // 3600

        for i in range(n_cells-order):
            context = '-'.join(cells[i:order+i])
            context_idx = ctxt_to_idx[context]
            true_next = cells[order+i]
            true_next_idx = cell_to_idx[true_next]
    
            hour = timestamps[order+i-1] # Heure du dernier record
            
            contexts.append(context_idx)
            hours.append(hour)
            targets.append(true_next_idx)
            
    samples = torch.tensor(list(zip(contexts, hours, targets)),dtype=torch.long)
    torch.save(samples, output_sample_path)
    
if __name__ == "__main__":
    MAIN_DIR = Path(__file__).parent.parent.parent
    
    cell_to_idx_path = MAIN_DIR / "results/predictions/deep_learning/cell_map.json"
    with open(cell_to_idx_path, mode='r', newline='') as f:
        cell_to_idx = json.load(f)
        
    orders = list(range(1,6))
    
    # ================================ #
    # Generate samples for train split #
    # ================================ # 
    TRAIN_CSV_PATH = MAIN_DIR / "results/predictions/train_test/train_random.csv"
    with open(TRAIN_CSV_PATH, mode='r', newline='') as f:
        reader = csv.reader(f, delimiter=";")
        users = list(reader)

    for order in orders:
        ctxt_to_idx_path = MAIN_DIR / f"results/predictions/mappings/ctxt_to_idx_{order}.json"
        with open(ctxt_to_idx_path, mode='r', newline='') as f:
            ctxt_to_idx = json.load(f)
        
        # Train samples
        output_sample_path = MAIN_DIR / f"results/predictions/train_test/train_samples_random_order_{order}.pt"
        main(users = users, cell_to_idx = cell_to_idx, ctxt_to_idx=ctxt_to_idx, order=order,output_sample_path=output_sample_path)


    # ============================== #
    # Generate samples for val split #
    # ============================== # 
    VAL_CSV_PATH = MAIN_DIR / "results/predictions/train_test/val_random.csv"
    with open(VAL_CSV_PATH, mode='r', newline='') as f:
        reader = csv.reader(f, delimiter=";")
        users = list(reader)

    for order in orders:
        ctxt_to_idx_path = MAIN_DIR / f"results/predictions/mappings/ctxt_to_idx_{order}.json"
        with open(ctxt_to_idx_path, mode='r', newline='') as f:
            ctxt_to_idx = json.load(f)
        
        # Validation samples
        output_sample_path = MAIN_DIR / f"results/predictions/train_test/val_samples_random_order_{order}.pt"
        main(users = users, cell_to_idx = cell_to_idx, ctxt_to_idx=ctxt_to_idx, order=order,output_sample_path=output_sample_path)
