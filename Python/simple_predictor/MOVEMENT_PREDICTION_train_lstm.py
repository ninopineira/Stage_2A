"""
train_lstm_movement.py
──────────────────────
Entraîne un LSTM sur les features V1 (y compris les features temporelles
à faible importance individuelle) pour capturer leurs interactions séquentielles.

Idée centrale : au lieu de passer UN vecteur de features à t=T,
on passe la SÉQUENCE des vecteurs [t=1, t=2, ..., t=T] au LSTM.
Chaque pas de temps = les features extraites sur l'historique tronqué jusqu'à ce point.

Cela permet au modèle de voir comment first_departure_elapsed_h ÉVOLUE,
comment local_acceleration CHANGE, etc. — le signal est dans la dérivée,
pas dans la valeur absolue.
"""

import json
import time
import random
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MAIN_DIR    = Path(__file__).parent.parent.parent
FEATURE_DIR = MAIN_DIR / "results/predictions/movement_prediction/features"
OUTPUT_DIR  = MAIN_DIR / "results/predictions/movement_prediction_lstm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR   = OUTPUT_DIR / "models_lstm"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = FEATURE_DIR / "class1_train_random_features_V4.csv"
TEST_CSV  = FEATURE_DIR / "class1_test_random_features_V4.csv"

# Features V1 — on garde TOUTES, y compris les temporelles à faible importance
# L'intérêt du LSTM est précisément de trouver les interactions entre elles
FEATURE_COLS = [
    # -- Temporelles (importance <0.01 en XGBoost isolé) --
    "timestamp-1", "timestamp-2", "timestamp-3",
    "time_diff-1", "time_diff-2",
    "mean_time_between_record",
    "first_departure_elapsed_h",
    "time_in_anchor_before_dep_h",
    "time_since_return_h",
    "current_trip_duration_h",
    # -- Structurelles (top-5 importance XGBoost) --
    "local_acceleration",
    "is_at_anchor_cell",
    "moving",
    "trip_phase",
    "returned_to_anchor",
    # -- Autres --
    "record_per_hour",
    "number_of_time_in_first_cell",
    "has_departed_today",
    "n_complete_trips",
    "out_of_anchor_entropy",
]
LABEL_COL  = "label"
N_FEATURES = len(FEATURE_COLS)

# Hyperparamètres
SEQ_LEN        = 8       # Nombre de pas de temps dans la séquence LSTM
                          # = on reconstruit l'historique sur les 8 derniers
                          # splits possibles pour chaque user
HIDDEN_SIZE    = 128
N_LAYERS       = 2
DROPOUT        = 0.3
BATCH_SIZE     = 2048
EPOCHS         = 5
LR             = 1e-3
POS_WEIGHT_FACTOR = 1.5  # Moins agressif que 3.33 (voir analyse précision)
THRESHOLD      = 0.50    # Affiné après première run via find_best_threshold()
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device : {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# DATASET — reconstruction de séquences depuis le CSV de features
# ══════════════════════════════════════════════════════════════════════════════

class SequenceMovementDataset(Dataset):
    """
    Transforme le CSV de features (1 ligne = 1 snapshot à un instant t)
    en séquences de longueur SEQ_LEN.

    Problème : le CSV actuel contient des snapshots indépendants (random_split).
    Solution : on regroupe les features par blocs de SEQ_LEN lignes consécutives
    du CSV — c'est une approximation valide car les lignes sont tirées
    aléatoirement depuis des users différents, donc chaque ligne est
    indépendante. Pour une vraie séquence par user il faudrait re-extraire
    (voir rebuild_sequences_from_raw()).

    Pour exploiter pleinement le LSTM, utilise rebuild_sequences_from_raw()
    qui reconstruit de vraies séquences temporelles par user.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.seq_len = seq_len
        # Découper en séquences de longueur fixe
        # On tronque pour avoir un multiple exact de seq_len
        n_complete = (len(X) // seq_len) * seq_len
        X = X[:n_complete]
        y = y[:n_complete]
        # Reshape : (N, seq_len, n_features)
        self.X = torch.tensor(
            X.reshape(-1, seq_len, X.shape[1]),
            dtype=torch.float32
        )
        # Label = label du DERNIER pas de la séquence
        self.y = torch.tensor(
            y.reshape(-1, seq_len)[:, -1],
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TrueSequenceDataset(Dataset):
    """
    Dataset avec de VRAIES séquences par utilisateur.
    Chaque séquence = [features@t1, features@t2, ..., features@tT]
    où t1 < t2 < ... < tT pour le MÊME utilisateur le MÊME jour.

    C'est le dataset idéal pour le LSTM — à utiliser si tu re-extraits
    depuis les CSVs bruts avec rebuild_sequences_from_raw().
    """

    def __init__(self, sequences: list[np.ndarray], labels: list[int]):
        """
        sequences : liste de tableaux (seq_len, n_features)
        labels    : label final de chaque séquence (0 ou 1)
        """
        assert len(sequences) == len(labels)
        self.X = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def rebuild_sequences_from_raw(
    raw_csv_path: Path,
    extractor,          # instance de MovementPredictor
    seq_len: int = SEQ_LEN,
    min_records: int = 10,
    max_records: int = 512,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Re-extrait des VRAIES séquences temporelles depuis le CSV brut.

    Pour chaque user avec assez de records, on génère une séquence de
    `seq_len` snapshots de features espacés régulièrement dans l'historique :
        t1 = split à 30% de l'historique
        t2 = split à 40%
        ...
        tT = split aléatoire final (le point de prédiction réel)

    Le label est celui du split final.

    Returns:
        sequences : list of (seq_len, n_features) arrays
        labels    : list of int (0 ou 1)
    """
    sequences, labels = [], []

    with open(raw_csv_path, mode="r", newline='') as f:
        reader = csv.reader(f, delimiter=";")
        for user in reader:
            n_preds = int(user[7])
            if not (min_records < n_preds < max_records):
                continue

            cells      = user[8::2]
            timestamps = [int(ts) for ts in user[9::2]]

            # Point de prédiction final aléatoire
            final_split = random.randint(min_records, n_preds - 1)
            true_next   = cells[final_split]
            label       = int(cells[final_split - 1] != true_next)

            # Construire la séquence de seq_len snapshots
            # Les splits intermédiaires sont espacés régulièrement
            # entre min_records et final_split
            if final_split < seq_len + 2:
                continue  # Pas assez d'historique

            split_points = np.linspace(
                max(5, final_split - seq_len * 3),
                final_split,
                seq_len,
                dtype=int,
            )
            # Dédoublonner au cas où linspace génère des égaux
            split_points = sorted(set(split_points.tolist()))
            if len(split_points) < seq_len:
                # Compléter par le début si pas assez de points distincts
                split_points = list(range(5, 5 + seq_len))
                if max(split_points) >= final_split:
                    continue

            seq_features = []
            for sp in split_points[-seq_len:]:
                h_cells  = cells[:sp]
                h_ts     = timestamps[:sp]
                feat_dict = extractor.extract_features(h_cells, h_ts)
                feat_vec  = [feat_dict[col] for col in FEATURE_COLS]
                seq_features.append(feat_vec)

            sequences.append(np.array(seq_features, dtype=np.float32))
            labels.append(label)

    print(f"  Rebuilt {len(sequences):,} true sequences from {raw_csv_path.name}")
    return sequences, labels


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLES
# ══════════════════════════════════════════════════════════════════════════════

class MovementLSTM(nn.Module):
    """
    LSTM bidirectionnel + tête de classification.

    Architecture :
        Input  (batch, seq_len, n_features)
            ↓
        BatchNorm1d sur les features (normalise chaque feature indépendamment)
            ↓
        LSTM bidirectionnel (hidden_size, n_layers, dropout)
            ↓
        Prend le dernier hidden state (avant + arrière concaténés)
            ↓
        Dropout + Linear(hidden*2 → 64) + GELU
            ↓
        Linear(64 → 1) + Sigmoid
            ↓
        Output (batch,) — probabilité de mouvement

    Pourquoi bidirectionnel ?
        Le LSTM forward lit [t1→tT] et capture "l'accélération vers le départ".
        Le LSTM backward lit [tT→t1] et capture "la décélération vers le retour".
        Les deux ensemble donnent le contexte complet du mouvement.
    """

    def __init__(
        self,
        n_features:  int,
        hidden_size: int = HIDDEN_SIZE,
        n_layers:    int = N_LAYERS,
        dropout:     float = DROPOUT,
    ):
        super().__init__()

        # Normalisation des features en entrée
        self.input_norm = nn.BatchNorm1d(n_features)

        # LSTM bidirectionnel
        self.lstm = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_size,
            num_layers   = n_layers,
            batch_first  = True,
            dropout      = dropout if n_layers > 1 else 0.0,
            bidirectional= True,
        )

        # Tête de classification
        lstm_out_size = hidden_size * 2  # × 2 car bidirectionnel
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, n_features)
        Returns : (batch,) probabilités ∈ [0, 1]
        """
        # BatchNorm attend (batch, features) ou (batch, features, seq)
        # On normalise sur la dimension features pour chaque pas de temps
        batch, seq, feat = x.shape
        x = x.view(batch * seq, feat)
        x = self.input_norm(x)
        x = x.view(batch, seq, feat)

        # LSTM : output shape (batch, seq_len, hidden*2)
        lstm_out, (h_n, _) = self.lstm(x)

        # Prendre le dernier pas de temps (le plus récent)
        # Pour un LSTM bidirectionnel : h_n shape = (n_layers*2, batch, hidden)
        # On concatène le dernier forward et backward hidden state
        h_forward  = h_n[-2]  # dernier layer, direction forward
        h_backward = h_n[-1]  # dernier layer, direction backward
        h_last = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden*2)

        return self.head(h_last).squeeze(1)  # (batch,)


class MovementMLP(nn.Module):
    """
    MLP profond avec skip connections (style ResNet).
    Alternative au LSTM pour capturer les interactions non-linéaires
    entre features sans modéliser l'ordre temporel.

    Utile comme baseline "interactions sans séquence" pour comparer
    avec le LSTM "interactions + séquence".
    """

    def __init__(self, n_features: int, dropout: float = DROPOUT):
        super().__init__()

        self.input_norm = nn.BatchNorm1d(n_features)

        # Bloc d'entrée
        self.entry = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
        )

        # Blocs résiduels (permettent au gradient de circuler profondément)
        self.res1 = self._residual_block(256)
        self.res2 = self._residual_block(256)
        self.res3 = self._residual_block(256)

        # Tête de sortie
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _residual_block(size: int, dropout: float = DROPOUT) -> nn.Module:
        return nn.Sequential(
            nn.Linear(size, size),
            nn.GELU(),
            nn.BatchNorm1d(size),
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, n_features)  →  (batch,)"""
        x = self.input_norm(x)
        x = self.entry(x)
        x = x + self.res1(x)   # skip connection
        x = x + self.res2(x)
        x = x + self.res3(x)
        return self.head(x).squeeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    is_lstm:   bool,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if not is_lstm:
            # MLP : aplatir la séquence en vecteur si besoin
            if X_batch.dim() == 3:
                X_batch = X_batch.view(X_batch.size(0), -1)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model:    nn.Module,
    loader:   DataLoader,
    criterion: nn.Module,
    device:   torch.device,
    is_lstm:  bool,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if not is_lstm and X_batch.dim() == 3:
            X_batch = X_batch.view(X_batch.size(0), -1)

        probs = model(X_batch)
        loss  = criterion(probs, y_batch)
        total_loss += loss.item() * len(y_batch)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels, dtype=int)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "loss":          total_loss / len(loader.dataset),
        "f1":            f1_score(y_true, y_pred, zero_division=0),
        "f1_macro":      f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision":     precision_score(y_true, y_pred, zero_division=0),
        "recall":        recall_score(y_true, y_pred, zero_division=0),
        "roc_auc":       roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "y_prob":        y_prob,
        "y_true":        y_true,
        "y_pred":        y_pred,
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Cherche le seuil θ qui maximise le F1 sur le set de validation.
    À appeler après l'entraînement, avant l'évaluation finale sur test.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.3, 0.8, 51):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"  Best threshold : {best_t:.2f}  (F1 = {best_f1:.4f})")
    return best_t


def run_training(
    model:       nn.Module,
    model_name:  str,
    train_loader: DataLoader,
    val_loader:  DataLoader,
    test_loader: DataLoader,
    is_lstm:     bool,
    pos_weight:  float,
):
    print(f"\n{'═'*55}")
    print(f"  Training {model_name}")
    print(f"{'═'*55}")

    model = model.to(DEVICE)
    criterion = nn.BCELoss(
        weight=None  # pos_weight géré via oversampling ou threshold
    )
    # On utilise BCEWithLogitsLoss + pos_weight si on veut garder
    # le déséquilibre compensé côté loss (alternative au threshold)
    criterion_weighted = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_f1   = 0.0
    best_state    = None
    history       = []
    patience_left = 8  # early stopping

    for epoch in range(1, EPOCHS + 1):
        t0        = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE, is_lstm
        )
        val_metrics = evaluate(
            model, val_loader, criterion, DEVICE, is_lstm, THRESHOLD
        )

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:02d}/{EPOCHS}  "
            f"loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}  "
            f"val_prec={val_metrics['precision']:.4f}  "
            f"val_rec={val_metrics['recall']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()
               if k not in ("y_prob", "y_true", "y_pred")},
        })

        scheduler.step(val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 8
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restaurer le meilleur état
    model.load_state_dict(best_state)

    # Trouver le seuil optimal sur la validation
    val_final = evaluate(model, val_loader, criterion, DEVICE, is_lstm, 0.5)
    best_thresh = find_best_threshold(val_final["y_true"], val_final["y_prob"])

    # Évaluation finale sur test
    test_metrics = evaluate(
        model, test_loader, criterion, DEVICE, is_lstm, best_thresh
    )

    print(f"\n  ┌─ {model_name} — test metrics ─────────────────────────")
    for k, v in test_metrics.items():
        if k not in ("y_prob", "y_true", "y_pred"):
            print(f"  │  {k:<20} {v:.4f}")
    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"])
    tn, fp, fn, tp = cm.ravel()
    print(f"  │  TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")
    print(f"  └────────────────────────────────────────────────")
    print(f"\n{classification_report(test_metrics['y_true'], test_metrics['y_pred'], target_names=['Stay','Move'])}")

    # Sauvegardes
    torch.save(model.state_dict(), MODEL_DIR / f"{model_name.lower()}.pt")
    with open(MODEL_DIR / f"{model_name.lower()}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved → {model_name.lower()}.pt")

    return model, test_metrics, best_thresh


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n══ Movement LSTM/MLP — train & evaluate ══\n")

    # ── Chargement des features V1 ────────────────────────────────────────────
    print("► Loading feature CSVs …")

    def load_df(path: Path) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path, sep=";").dropna(subset=FEATURE_COLS + [LABEL_COL])
        X  = df[FEATURE_COLS].values.astype(np.float32)
        y  = df[LABEL_COL].values.astype(np.float32)
        print(f"  {path.name} : {len(y):,} samples  |  move rate = {y.mean():.3f}")
        return X, y

    X_train_raw, y_train_raw = load_df(TRAIN_CSV)
    X_test_raw,  y_test_raw  = load_df(TEST_CSV)

    # ── Normalisation (StandardScaler fitted sur train uniquement) ────────────
    print("\n► Normalizing features …")
    mean = X_train_raw.mean(axis=0)
    std  = X_train_raw.std(axis=0) + 1e-8  # éviter division par 0

    X_train_norm = (X_train_raw - mean) / std
    X_test_norm  = (X_test_raw  - mean) / std

    # Sauvegarder les stats de normalisation pour l'inférence
    np.save(MODEL_DIR / "norm_mean.npy", mean)
    np.save(MODEL_DIR / "norm_std.npy",  std)

    # ── Split train/val (90/10) ───────────────────────────────────────────────
    n_val   = int(0.1 * len(X_train_norm))
    indices = np.random.permutation(len(X_train_norm))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_tr, y_tr = X_train_norm[train_idx], y_train_raw[train_idx]
    X_val, y_val = X_train_norm[val_idx], y_train_raw[val_idx]

    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum()) * POS_WEIGHT_FACTOR

    print(f"  Train: {len(X_tr):,}  |  Val: {len(X_val):,}  |  pos_weight={pos_weight:.2f}")

    # ── DataLoaders LSTM (séquences) ──────────────────────────────────────────
    print(f"\n► Building sequence datasets (seq_len={SEQ_LEN}) …")
    lstm_train_ds = SequenceMovementDataset(X_tr,   y_tr,  SEQ_LEN)
    lstm_val_ds   = SequenceMovementDataset(X_val,  y_val, SEQ_LEN)
    lstm_test_ds  = SequenceMovementDataset(X_test_norm, y_test_raw, SEQ_LEN)

    lstm_train_loader = DataLoader(lstm_train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    lstm_val_loader   = DataLoader(lstm_val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    lstm_test_loader  = DataLoader(lstm_test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ── DataLoaders MLP (vecteurs plats) ─────────────────────────────────────
    # Pour le MLP on utilise les mêmes séquences aplaties
    # (seq_len * n_features) comme vecteur d'entrée
    mlp_n_features = SEQ_LEN * N_FEATURES

    mlp_train_loader = DataLoader(lstm_train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    mlp_val_loader   = DataLoader(lstm_val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    mlp_test_loader  = DataLoader(lstm_test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRAÎNEMENT LSTM
    # ══════════════════════════════════════════════════════════════════════════

    lstm_model = MovementLSTM(
        n_features  = N_FEATURES,
        hidden_size = HIDDEN_SIZE,
        n_layers    = N_LAYERS,
        dropout     = DROPOUT,
    )
    print(f"\n  LSTM params : {sum(p.numel() for p in lstm_model.parameters()):,}")

    lstm_model, lstm_metrics, lstm_thresh = run_training(
        model        = lstm_model,
        model_name   = "LSTM",
        train_loader = lstm_train_loader,
        val_loader   = lstm_val_loader,
        test_loader  = lstm_test_loader,
        is_lstm      = True,
        pos_weight   = pos_weight,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ENTRAÎNEMENT MLP
    # ══════════════════════════════════════════════════════════════════════════

    mlp_model = MovementMLP(n_features=mlp_n_features, dropout=DROPOUT)
    print(f"\n  MLP params : {sum(p.numel() for p in mlp_model.parameters()):,}")

    mlp_model, mlp_metrics, mlp_thresh = run_training(
        model        = mlp_model,
        model_name   = "MLP",
        train_loader = mlp_train_loader,
        val_loader   = mlp_val_loader,
        test_loader  = mlp_test_loader,
        is_lstm      = False,
        pos_weight   = pos_weight,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # RÉCAPITULATIF
    # ══════════════════════════════════════════════════════════════════════════

    print("\n══ Résumé comparatif ══\n")
    print(f"{'Métrique':<20} {'XGBoost V1':>12} {'LSTM':>12} {'MLP':>12}")
    print("─" * 58)
    xgb_ref = {"precision": 0.4762, "recall": 0.8005, "f1": 0.5972, "roc_auc": 0.8413}
    for metric in ("precision", "recall", "f1", "roc_auc"):
        xgb_val  = xgb_ref.get(metric, 0)
        lstm_val = lstm_metrics.get(metric, 0)
        mlp_val  = mlp_metrics.get(metric, 0)
        print(f"{metric:<20} {xgb_val:>12.4f} {lstm_val:>12.4f} {mlp_val:>12.4f}")

    # Sauvegarde du résumé
    summary = {
        "XGBoost_V1": xgb_ref,
        "LSTM":  {k: float(v) for k, v in lstm_metrics.items() if k not in ("y_prob","y_true","y_pred")},
        "MLP":   {k: float(v) for k, v in mlp_metrics.items()  if k not in ("y_prob","y_true","y_pred")},
    }
    with open(OUTPUT_DIR / "lstm_vs_mlp_vs_xgb.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n══ Done ══\n")


if __name__ == "__main__":
    main()