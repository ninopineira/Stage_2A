# Démo / débogage : prédit la prochaine cellule pour UNE séquence donnée, et affiche
# les k meilleurs candidats avec leur probabilité.
#
# Utile pour inspecter le comportement du VOMM sur un contexte précis (par exemple pour
# vérifier qu'il ne se contente pas de répéter la dernière cellule du contexte) sans
# lancer tout le pipeline d'évaluation.
#
# Prérequis (pipeline dataset_creation) :
#   1_train_test_split.py  ->  2b_create_train_ngrams_matrix.py

import json
import time
from pathlib import Path

from models import VOMM

MAIN_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = MAIN_DIR / "Database/no_duplicate"

TRAINING_MATRIX_DIR = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix"
TRAIN_DATA_PATH = TRAINING_MATRIX_DIR / "ngrams_matrix_train_random.json"
CTX_TOTALS_PATH = TRAINING_MATRIX_DIR / "context_totals_train.json"
UNIQUE_PATH     = TRAINING_MATRIX_DIR / "unique_followers_train.json"
UNIGRAM_PATH    = TRAINING_MATRIX_DIR / "unigram_train.json"

MAX_CONTEXT = 5
DISCOUNT    = 0.9
TOP_K       = 10

# La séquence à prolonger (de la plus ancienne à la plus récente).
CONTEXT = ["BKVRUP4", "BKVRUP4", "BKVMEZ1", "BKVMEZ1", "BKVMEZ1"]


if not TRAIN_DATA_PATH.exists():
    raise FileNotFoundError(
        f"{TRAIN_DATA_PATH} introuvable.\nLancer d'abord le pipeline dataset_creation : "
        "1_train_test_split.py -> 2b_create_train_ngrams_matrix.py"
    )

print("================================")
print("Preparing data for prediction...")
print("================================")
t0 = time.time()

with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)

# Les 3 fichiers de features (totaux par contexte, suffixes uniques, unigrammes) sont
# relus s'ils existent, sinon calculés puis sauvegardés par le constructeur.
model = VOMM(max_order=MAX_CONTEXT, discount=DISCOUNT,
             train_data=train_data,
             context_totals_train_data_path=CTX_TOTALS_PATH,
             unique_train_data_path=UNIQUE_PATH,
             unigram_train_data_path=UNIGRAM_PATH,
             DATASET_DIR=DATASET_DIR)

print("================================")
print(f"Data prepared in {time.time() - t0:.2f} seconds")
print("================================")


# ========== #
# Prediction #
# ========== #
print("================================")
print(f"Predicting next cell for the sequence : \n{CONTEXT}")
print("================================")
t0 = time.time()

# predict_next renvoie TOUS les candidats triés par probabilité décroissante :
# [(cellule, probabilité), ...]
scores = model.predict_next(CONTEXT)

print(f"Prediction done in {time.time() - t0:.2f}s")
print(f"{len(scores)} candidates scored — top {TOP_K} :\n")
print(f"  {'rang':<6}{'cellule':<14}{'probabilité'}")
print("  " + "-" * 36)
for rank, (cell, prob) in enumerate(scores[:TOP_K], start=1):
    marker = "  <- même que la dernière du contexte" if cell == CONTEXT[-1] else ""
    print(f"  {rank:<6}{cell:<14}{prob:.6f}{marker}")
