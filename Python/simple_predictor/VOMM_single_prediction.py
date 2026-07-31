import time
from pathlib import Path
from models import VOMM
from utils import HelperVOMM

MAIN_DIR = Path(__file__).parent.parent.parent

DATASET_DIR = MAIN_DIR / f"Database/no_duplicate"
CLASSIFICATION_PATH = MAIN_DIR / f"results/cd_142/intermediate_result/cell_classification.csv"
N_GRAM_PATH = MAIN_DIR / "results/simple_predictor/transition_matrix/ngrams/transition_ngrams_number.json"


helper = HelperVOMM()
t0 = time.time()
print("================================")
print("Preparing data for prediction...")
print("================================")
ngrams_counts, contexts, unique = helper.prepare_ngrams(computed_ngrams_filepath = N_GRAM_PATH)
unigram_counts = helper.prepare_unigram_counts(dataset_filepath = DATASET_DIR)

model = VOMM(max_order=5, discount=0.75, 
             counts=ngrams_counts, context_totals=contexts, unique_followers=unique,
             unigram_counts=unigram_counts)
tf = time.time() - t0
print("================================")
print(f"Data prepared in {tf:.2f} seconds")
print("================================")


# ========== #
# Prediction #
# ========== #
context = ["BKVRUP4","BKVRUP4","BKVMEZ1","BKVMEZ1","BKVMEZ1"]
print("================================")
print(f"Predicting next cell for the sequence : \n{context}")
print("================================")
t0 = time.time()

preds, top_preds = model.predict_next(context, top_k=10)

tf = time.time() - t0
print("================================")
print(f"Prediction done in {tf:.2f}")
print("================================")


