# ~27mins/epoch


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import MobilityTransformer
from dataset import MobilityDataset

import argparse
import json
from pathlib import Path
import tqdm
# ---------------------------------------------------------------------------
# Constantes de padding / tokens spéciaux
# ---------------------------------------------------------------------------
 
PAD_CELL_ID = 0      # index réservé — jamais une vraie cellule
EOS_OFFSET  = 1      # EOS_CELL_ID = n_real_cells + 1  (calculé dynamiquement)
# Index 0          → padding (ignoré dans la loss)
# Index 1..N       → vraies cellules
# Index N+1        → EOS (fin de séquence)
MAX_SEQ_LEN  = 512   # troncature à droite
MIN_CONTEXT  = 6     # on ne prédit pas les MIN_CONTEXT premiers tokens



# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------
 
def train_one_epoch(
    model: MobilityTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    min_context: int = MIN_CONTEXT,
) -> float:
    """
    Stratégie teacher-forcing décalée d'un pas :
        input  : tokens 0 .. L-2
        cible  : tokens 1 .. L-1
 
    On ne calcule la loss que pour les positions ≥ min_context et non-paddées,
    afin d'avoir un contexte minimal avant la première prédiction.
 
    Retourne la loss moyenne sur l'époque.
    """
    model.train()
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_CELL_ID,   # ignore les positions paddées dans la cible
        reduction="mean",
    )
    total_loss = 0.0
    n_batches  = 0
 
    for cells, times, mask in tqdm.tqdm(loader, desc=f"Training epoch {epoch}", colour="red"):
        cells = cells.long().to(device)   # (B, L)
        times = times.to(device)
        mask  = mask.to(device)
 
        # Décalage d'un pas (teacher-forcing)
        inp_cells = cells[:, :-1]      # (B, L-1)
        inp_times = times[:, :-1]
        inp_mask  = mask[:, :-1]
        tgt_cells = cells[:, 1:]       # (B, L-1)  ← cible
 
        logits = model(inp_cells, inp_times, inp_mask)   # (B, L-1, vocab)
 
        # Masque : on prédit seulement à partir de la position min_context
        # et seulement sur des positions non-paddées dans la CIBLE
        pred_mask = ~inp_mask                             # True = position réelle
        pred_mask[:, :min_context] = False                # ignore les < min_context premières
 
        # Aplatissement pour CrossEntropyLoss
        # On met PAD_CELL_ID sur les positions masquées → ignore_index les élimine
        tgt_masked = tgt_cells.clone()
        tgt_masked[~pred_mask] = PAD_CELL_ID
 
        B, L, V = logits.shape
        loss = criterion(logits.reshape(B * L, V), tgt_masked.reshape(B * L))
 
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
 
        total_loss += loss.item()
        n_batches  += 1
 
    return total_loss / max(n_batches, 1)
 
 
@torch.no_grad()
def evaluate(
    model: MobilityTransformer,
    loader: DataLoader,
    device: torch.device,
    min_context: int = MIN_CONTEXT,
) -> dict:
    """Calcule loss et accuracy top-1 sur le loader fourni."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_CELL_ID, reduction="sum")
 
    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0
 
    for cells, times, mask in tqdm.tqdm(loader, desc=f"Evaluation for epoch {epoch}", colour="yellow"):
        cells = cells.long().to(device)
        times = times.to(device)
        mask  = mask.to(device)
 
        inp_cells = cells[:, :-1]
        inp_times = times[:, :-1]
        inp_mask  = mask[:, :-1]
        tgt_cells = cells[:, 1:]
 
        logits = model(inp_cells, inp_times, inp_mask)
 
        pred_mask = ~inp_mask
        pred_mask[:, :min_context] = False
 
        tgt_masked = tgt_cells.clone()
        tgt_masked[~pred_mask] = PAD_CELL_ID
 
        B, L, V = logits.shape
        total_loss += criterion(logits.reshape(B * L, V), tgt_masked.reshape(B * L)).item()
 
        preds   = logits.argmax(dim=-1)          # (B, L)
        correct = (preds == tgt_cells) & pred_mask
        total_correct += correct.sum().item()
        total_tokens  += pred_mask.sum().item()
 
    n = max(total_tokens, 1)
    return {
        "loss":     total_loss / n,
        "accuracy": total_correct / n,
        "n_tokens": total_tokens,
    }
 


def prepare_training(train_data_path, val_data_path, save_path):
    
    
    train_ds = MobilityDataset(sample_path = train_data_path)
    val_ds   = MobilityDataset(sample_path = val_data_path  )
 
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
 
    # ── From checkpoint or not ──────────────────────
    RESUME_FROM = save_path
    start_epoch = 1

    if RESUME_FROM is not None and Path(RESUME_FROM).exists():
        print(f"\nChargement du checkpoint : {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=DEVICE)

        # Vérification de cohérence des hyperparamètres
        for key in ("n_cells", "d_model", "n_heads", "n_layers", "d_ff"):
            saved_val   = ckpt.get(key)
            current_val = {"n_cells": N_CELLS, "d_model": D_MODEL,
                        "n_heads": N_HEADS, "n_layers": N_LAYERS,
                        "d_ff": D_FF}[key]
            if saved_val is not None and saved_val != current_val:
                raise ValueError(
                    f"Hyperparamètre '{key}' incohérent : "
                    f"checkpoint={saved_val}, script={current_val}"
                )

        model.load_state_dict(ckpt["model_state"])

        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])

        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Reprise à partir de l'époque {start_epoch}  "
            f"(checkpoint sauvegardé après époque {start_epoch - 1})")
    else:
        # ── Modèle ────────────────────────────────────────────────────────────
        model = MobilityTransformer(
            n_cells  = N_CELLS,
            d_model  = D_MODEL,
            n_heads  = N_HEADS,
            n_layers = N_LAYERS,
            d_ff     = D_FF,
            max_seq_len = MAX_SEQ_LEN,
            pad_cell_id = PAD_CELL_ID,
            dropout  = DROPOUT,
        ).to(DEVICE)
    
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Paramètres entraînables : {n_params:,}")
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LR,
            steps_per_epoch=len(train_loader),
            epochs=N_EPOCHS,
            pct_start=0.1,
        )
        print("\nNo checkpoint found — training from scratch.")

    return model, optimizer, scheduler, train_loader, val_loader, start_epoch


# ---------------------------------------------------------------------------
# Point d'entrée principal — test rapide en local
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    import time

    parser = argparse.ArgumentParser(description="model")
    parser.add_argument('--save-path', type=str, help="Path of the save of the model | If no model to load, keep it unused", required=False, default=None)
    args = vars(parser.parse_args())
    save_path = args["save_path"]

    MAIN_DIR = Path(__file__).parent.parent.parent
    OUTPUT_DIR = MAIN_DIR / "results/predictions/deep_learning"

    # N_CELLS est lu depuis le mapping produit par create_cellid_map.py, et non codé en
    # dur : c'est ce même mapping qui a servi à encoder les .pt, donc les deux ne peuvent
    # pas diverger (une valeur codée en dur trop petite tronque le vocabulaire et fait
    # planter l'embedding sur les cellules d'indice élevé).
    CELL_MAP_PATH = OUTPUT_DIR / "cell_map_start_1.json"
    if not CELL_MAP_PATH.exists():
        raise FileNotFoundError(
            f"{CELL_MAP_PATH} introuvable — lancer d'abord create_cellid_map.py "
            "puis encode_and_convert_csv_to_pytorch.py."
        )
    with open(CELL_MAP_PATH, mode="r", encoding="utf-8") as f:
        N_CELLS = len(json.load(f))   # nombre de cellules réelles (indices 1..N)

    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_MODEL   = 128     # 128 → 512 (×4, impact quadratique sur l'attention)
    N_HEADS   = 4       # 4   → 8
    N_LAYERS  = 2       # 2   → 6    (le levier le plus puissant sur les params)
    D_FF      = 512     # 512 → 2048 (convention : 4×d_model)
    BATCH     = 32      # 8   → 128  (remplit la VRAM avec les grandes séquences 512)
    N_EPOCHS  = 3
    LR        = 1e-4    # à baisser un peu avec un modèle plus grand
    DROPOUT   = 0.1
 
    MAIN_DIR = Path(__file__).parent.parent.parent
 
    print(f"Device : {DEVICE}")
    print(f"Cellules : {N_CELLS}  |  vocab_size : {N_CELLS + 2} (CELLS + EOS and PADDING)")
 
    print("\nData prep for training...")
    train_data_path = MAIN_DIR / "results/predictions/deep_learning/encoded_train.pt"
    val_data_path   = MAIN_DIR / "results/predictions/deep_learning/encoded_val.pt"
    model, optimizer, scheduler, train_loader, val_loader, start_epoch = prepare_training(train_data_path = train_data_path, 
                                                                             val_data_path = val_data_path, 
                                                                             save_path = save_path)
    print("Everything loaded, starting training...")
 
    # ── Boucle d'entraînement ─────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, N_EPOCHS + 1):
            t0 = time.time()
    
            train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
            scheduler.step()   # OneCycleLR se met à jour par step mais on appelle ici pour simplifier
    
            val_metrics = evaluate(model, val_loader, DEVICE)
            elapsed     = time.time() - t0
    
            print(
                f"Epoch {epoch}/{N_EPOCHS} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.3%} | "
                f"tokens={val_metrics['n_tokens']:,} | "
                f"{elapsed:.1f}s"
            )
    except KeyboardInterrupt:
        print(f"\nTest aborted at epoch {epoch}")
        # ── Sauvegarde ────────────────────────────────────────────────────────
        output_path = (OUTPUT_DIR / "checkpoints")
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"tupe_mobility_epoch_{epoch}.pt"
        save_path.parent.mkdir(exist_ok=True)
        torch.save({
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch":           epoch,          
            "n_cells":         N_CELLS,
            "d_model":         D_MODEL,
            "n_heads":         N_HEADS,
            "n_layers":        N_LAYERS,
            "d_ff":            D_FF,
        }, save_path)
        print(f"Modèle sauvegardé → {save_path}")
        raise KeyboardInterrupt()
 
    print("\nTest terminé. Si val_loss < train_loss_epoch_1, le modèle apprend.")
    
    # ── Sauvegarde ────────────────────────────────────────────────────────
    output_path = (OUTPUT_DIR / "checkpoints")
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / "tupe_mobility.pt"
    save_path.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch":           epoch,          # dernière époque complétée
        "n_cells":         N_CELLS,
        "d_model":         D_MODEL,
        "n_heads":         N_HEADS,
        "n_layers":        N_LAYERS,
        "d_ff":            D_FF,
    }, save_path)
    print(f"Modèle sauvegardé → {save_path}")














