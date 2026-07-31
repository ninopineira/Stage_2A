# This scripts aims to learn time weights for prediction models with a gradient descent algorithm

import csv
import json
import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from utils import MobilityDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class TimeWeightOptimizer(nn.Module):

    def __init__(self, num_hours=24, num_cells=369):
        super().__init__()

        # paramètres entraînables
        self.theta = nn.Parameter(
            torch.zeros(num_hours, num_cells)
        )

    def get_weights(self):
        # softmax sur les cellules pour chaque heure
        return torch.exp(self.theta) # AVANT :  torch.softmax(theta, dim=1) --> loss super stable = bof



def main(train_dataset : MobilityDataset, val_dataset : MobilityDataset,
         learning_rate : float, markov_matrix,
         order : int, output_path_weight : Path):
    
    # Weight "model" to optimize
    model = TimeWeightOptimizer(num_hours=24, num_cells=len(all_cells))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    epochs = 5
    
    # Send some data to GPU
    device = "cuda"
    model.to(device)
    markov_matrix = markov_matrix.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=True,
        pin_memory=True
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        
        train_loss = 0.0
        nb_train_batches = 0
                
        # ===== #
        # TRAIN #
        # ===== #
        for contexts, hours, targets in tqdm.tqdm(train_loader, desc=f"Training epoch {epoch+1}/{epochs}...", colour="red"):
            contexts = contexts.to(device)
            hours = hours.to(device)
            targets = targets.to(device)

            weights = model.get_weights()

            base_probs = markov_matrix[contexts]

            time_weights = weights[hours]

            combined = base_probs * time_weights

            combined = combined / (combined.sum(dim=1, keepdim=True) + 1e-12 )

            # Récupère les probas calculées pour les targets
            true_probs = combined[torch.arange(targets.size(0), device=device),targets] 
            
            loss = -torch.log(true_probs + 1e-12).mean() # Log_likelihood loss

            optimizer.zero_grad() # Reset du gradient
            loss.backward()       # Calcul du nouveau gradient à partir des loss    
            optimizer.step()      # Step d'optimizer
            train_loss += loss.item() # .item to store only the data and detach the data of the graph
            nb_train_batches += 1

        train_loss /= nb_train_batches
        train_losses.append(train_loss)

        # ========== #
        # VALIDATION #
        # ========== #
        model.eval()
        val_loss = 0.0
        nb_val_batches = 0
        with torch.no_grad():
            for contexts, hours, targets in tqdm.tqdm(val_loader, desc=f"Validating epoch {epoch+1}/{epochs}...", colour="yellow"):

                contexts = contexts.to(device, non_blocking=True)
                hours = hours.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                weights = model.get_weights()
                base_probs = markov_matrix[contexts]
                time_weights = weights[hours]
                
                scores = torch.log(base_probs + 1e-12) + time_weights

                combined = torch.softmax(scores, dim=1)
                
                #combined = combined / (combined.sum(dim=1, keepdim=True) + 1e-12)

                true_probs = combined[ torch.arange(targets.size(0), device=device),  targets]
                loss = -torch.log(true_probs + 1e-12).mean()
                val_loss += loss.item()
                nb_val_batches += 1

        val_loss /= nb_val_batches
        print(f"Epoch {epoch+1} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f}")
        
        print("weight mean and weight std", weights.mean(), weights.std())
        
        with torch.no_grad():
            delta = (weights - 1.0).abs().mean()
            print("mean |w-1| :", delta.item())
                    
        val_losses.append(val_loss)

    torch.save({"weights" : model.get_weights().detach().cpu()}, 
               f=output_path_weight / f"time_weight_markov_order_{order}.pt")

    return train_losses, val_losses

if __name__ == "__main__":
    assert torch.cuda.is_available(), "A cuda device is required to run this code"
    
    print("Starting some loadings...")
    
    MAIN_DIR = Path(__file__).parent.parent.parent
    cell_to_idx_path = MAIN_DIR / "results/predictions/deep_learning/cell_map.json"
    with open(cell_to_idx_path, mode="r") as f:
        cell_to_idx = json.load(f)
    all_cells = list(cell_to_idx.keys())
    orders = [str(i) for i in range(2,6)]
    learning_rate = 1e-2
    
    for order in orders:
        
        print("Setting up train matrix and samples...")
        TRAIN_MATRIX_PATH = MAIN_DIR / f"results/predictions/simple_predictor/transition_matrix/training_matrix/indexed_train_random_order_{order}.pt"
        markov_matrix = torch.load(TRAIN_MATRIX_PATH)
        train_samples_path = MAIN_DIR / f"results/predictions/train_test/train_samples_random_order_{order}.pt"
        val_samples_path   = MAIN_DIR / f"results/predictions/train_test/val_samples_random_order_{order}.pt"
        train_dataset = MobilityDataset(samples_path  = train_samples_path)
        val_dataset   = MobilityDataset(samples_path  = val_samples_path  )
        
        print("Start training")
        output_path_weight = MAIN_DIR / f"results/predictions/simple_predictor/time_weight"
        train_losses, val_losses =  main(train_dataset = train_dataset, val_dataset = val_dataset, 
             learning_rate = learning_rate, markov_matrix=markov_matrix,
             order=order, output_path_weight=output_path_weight)
        
        
        plt.plot(train_losses, label="train", color="blue")
        plt.plot(val_losses, label="validation", color="orange")

        plt.xlabel("Epoch")
        plt.ylabel("Log-likelihood Loss")

        plt.legend()

        plt.savefig(f"apprentissage_{order}.png")
        plt.close()