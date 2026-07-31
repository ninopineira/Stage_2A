import json
from pathlib import Path
import torch
import tqdm

if __name__ == "__main__":
    MAIN_DIR = Path(__file__).parent.parent.parent
    train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/ngrams_matrix_train_random.json"
    with open(train_data_path, mode='r', newline='') as f:
        train_matrix = json.load(f)
    train_matrix : dict[int,dict]
    
    cell_to_idx_path = MAIN_DIR / "results/predictions/deep_learning/cell_map.json"
    with open(cell_to_idx_path, mode='r', newline='') as f:
        cell_to_idx = json.load(f)
    num_cells = len(cell_to_idx)
    
    for order,contexts in train_matrix.items(): # Save order separately
        
        train_matrix_with_idx = {}
        
        all_keys = contexts.keys()
        idx_to_ctxt = {i : str_context for i,str_context in enumerate(all_keys)}
        ctxt_to_idx = {v : k for k,v in idx_to_ctxt.items()}
        
        num_contexts = len(all_keys)
        markov_matrix = torch.zeros(
            (num_contexts, num_cells),
            dtype=torch.float32
        )
        
        for ctxt, suff in tqdm.tqdm(contexts.items(), desc=f"Commputing prob for all context of order {order}", colour="blue"):
            context_idx = ctxt_to_idx[ctxt]
            total = sum(suff.values())
            for cell, count in suff.items():
                cell_idx = cell_to_idx[cell]
                prob = count / total
                markov_matrix[context_idx, cell_idx] = prob
        
        out_idx_to_ctxt = MAIN_DIR / f"results/predictions/mappings/idx_to_ctxt_{order}.json"
        out_idx_to_ctxt.parent.mkdir(parents=True, exist_ok=True)
        out_ctxt_to_idx = MAIN_DIR / f"results/predictions/mappings/ctxt_to_idx_{order}.json"
        
        with open(out_idx_to_ctxt, mode='w') as f:
            json.dump(idx_to_ctxt, f, indent=2)
        with open(out_ctxt_to_idx, mode='w') as f:
            json.dump(ctxt_to_idx, f, indent=2)
            
        out_markov_matrix_path = MAIN_DIR / f"results/predictions/simple_predictor/transition_matrix/training_matrix/indexed_train_random_order_{order}.pt"
        torch.save(markov_matrix, out_markov_matrix_path)
