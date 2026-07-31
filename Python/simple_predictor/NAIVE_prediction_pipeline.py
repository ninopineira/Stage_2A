# Prediction are made with the VOMM predictor with a full pipeline computing the ngrams without using the user 
# from which the sequence has been extracted to make the prediction. (Including the sequence in the training set)

from pathlib import Path
import json
import csv
import tqdm
import pandas as pd
from utils import Metrics
from models import NaiveMarkovChain, VOMM
import time

MAIN_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = MAIN_DIR / "Database/no_duplicate"
OUTPUT_ACC_DIR = MAIN_DIR / "results/predictions/simple_predictor/metrics/naive"
OUTPUT_ACC_DIR.mkdir(parents=True, exist_ok=True)


metric_man = Metrics()

##############
# MODEL RUNS #
##############
t0 = time.time()
# ===============================================================================
# MODELS B1 -> B5 : V1 : say that the the next is the most seen (give the k most seen)
# ===============================================================================
def run_naive_models(test_user, add_str = ""):
    
    acc_res = {}
    top_ks = [1,3,5,10]
    CONTEXT_LENGTH = ["1","2","3","4","5"]
    # CONTEXT_LENGTH = ["5"]

    # ================== #
    # STATS CHECK OBJECT #
    # ================== #
    same_as_last_context_dict = {} # Counts the number of predictions where the suffix that must be predicted is the last context
    user_unable_to_compute = {}
    meta_res = {}
    

    # test the 5 context length separately with naive markov predictor
    for order_int,order_str in enumerate(CONTEXT_LENGTH, start=1):
        # ============== #
        # RESULTS OBJECT #
        # ============== #
        top_k_list = []
        true_next_list = []
        preds_list = []
        acc_res[order_str] = {}
        
        number_of_preds = 0
        same_as_last_context_dict[order_str] = {"right" : 0,
                                                "wrong" : 0}
        
        user_unable_to_compute[order_str] = []
        
        naive_model = NaiveMarkovChain(order=order_int, train_data=train_data)
        # Test users
        for user in tqdm.tqdm(test_user[:10000], desc="Running naive model for all test users for context length " + order_str):
            n_cells = int(user[7])
            if n_cells < 6:
                user_unable_to_compute[order_str].append(user[0])
            else:
                for i in range(n_cells-order_int): # 0 -> nc-order-1 ; order+i -> nc-1  car on ne prend jamais nc-1 ds le ctxt
                    cells = user[8::2]
                    context = '-'.join(cells[i:order_int+i])
                    last = cells[order_int+i-1]
                    
                    true_next = cells[order_int+i]
                    top_preds, preds = naive_model.predict_next(context=context,last=last,top_k=10)
                    if top_preds[0][0] == last:
                        if last == true_next: 
                            same_as_last_context_dict[order_str]["right"] += 1
                        else : same_as_last_context_dict[order_str]["wrong"] += 1

                    top_k_list.append(top_preds)
                    true_next_list.append(true_next)
                    preds_list.append(preds)
                    
                    number_of_preds += 1
                    
        for top in top_ks:
            acc_res[order_str][f"ACC@{top}"] = metric_man.top_k_accuracy(preds_list=preds_list, true_next=true_next_list, k = top)
            acc_res[order_str][f"MAP@{top}"] = metric_man.map_k(preds_list=preds_list, true_next=true_next_list, k = top)
        acc_res[order_str]["log-l_mean"] = metric_man.log_likelihood(preds_list=preds_list, true_next_list=true_next_list)


        meta_res[order_str] = {}
        meta_res[order_str]["same_as_last_context_right"] = same_as_last_context_dict[order_str]["right"] / number_of_preds
        meta_res[order_str]["same_as_last_context_wrong"] = same_as_last_context_dict[order_str]["wrong"] / number_of_preds
        meta_res[order_str]["same_as_last_context_tot"] = meta_res[order_str]["same_as_last_context_right"] + meta_res[order_str]["same_as_last_context_wrong"]
    
    for order,v in user_unable_to_compute.items():
        meta_res[order]["nb_user_no_compute"] = len(v)


    df = pd.DataFrame(acc_res)
    df.to_csv(OUTPUT_ACC_DIR / f"acc_user_naive_{add_str}.csv", sep=";")
    df = pd.DataFrame(meta_res)
    df.to_csv(OUTPUT_ACC_DIR / f"meta_res_{add_str}.csv", sep=";")

    with open(OUTPUT_ACC_DIR / f"acc_user_naive_{add_str}.json", mode="w") as f:
        json.dump(acc_res, f, indent=2)
        
    with open(OUTPUT_ACC_DIR / f"meta_res_{add_str}.json", mode="w") as f:
        json.dump(meta_res, f, indent=2)

def run_VOMM_model(train_data, test_user, add_str = "",
                   context_totals_train_data_path = None,
                   unique_train_data_path = None,
                   unigram_train_data_path = None):
    MAX_CONTEXT_LENGTH = 5
    

    acc_res = {}
    top_ks = [1,3,5,10]

    VOMM_model = VOMM(max_order=MAX_CONTEXT_LENGTH, discount=0.75,
                train_data=train_data, context_totals_train_data_path = context_totals_train_data_path,
                unique_train_data_path = unique_train_data_path,
                unigram_train_data_path = unigram_train_data_path,
                DATASET_DIR = DATASET_DIR)

    preds_list = []
    true_next_list = []
    
    # Test users
    for user in tqdm.tqdm(test_user, desc="Running VOMM model for all test users...", colour="red"):
        n_cells = int(user[7])
        if n_cells < MAX_CONTEXT_LENGTH+1:
            continue
        cells = user[8::2]
        for context_start in range(n_cells-MAX_CONTEXT_LENGTH): # 0 -> nc-order-1 ; order+i -> nc-1  car on ne prend jamais nc-1 ds le ctxt
            
            next_index = MAX_CONTEXT_LENGTH+context_start
            context = tuple(cells[context_start:next_index])

            true_next = cells[next_index]
            preds = VOMM_model.predict_next(context)

            true_next_list.append(true_next)
            preds_list.append(preds)
    
    VOMM_model = None
    train_data = None
    del VOMM_model
    del train_data
    
    for top in top_ks:
        acc_res[f"top_{top}"] = metric_man.top_k_accuracy(preds_list=preds_list , true_next=true_next_list, k = top)
    acc_res["log-l_mean"] = metric_man.log_likelihood(preds_list=preds_list, true_next_list=true_next_list)

    with open(OUTPUT_ACC_DIR / f"acc_all_user_VOMM_{add_str}.json", mode="w") as f:
        json.dump(acc_res, f, indent=2)






# ===============================================================================
# MODELS B1 -> B5 : V2 (test split without self to self transitions)
# ===============================================================================

TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/ngrams_matrix_train_random.json"
with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# # # RUN 1 : First split random 80/20 : every existing user
# TEST_PATH = MAIN_DIR / "results/predictions/train_test/test_random.csv"
# with open(TEST_PATH, 'r', encoding='utf-8') as f:
#     test_users = csv.reader(f, delimiter=";")
#     test_users = list(test_users)
# run_naive_models(test_user=test_users, add_str="all")     


# RUN 2 : First split random 80/20  : only movements
TEST_PATH = MAIN_DIR / "results/predictions/train_test/removed_repeat/test_random.csv"
with open(TEST_PATH, 'r', encoding='utf-8') as f:
    test_users = csv.reader(f, delimiter=";")
    test_users = list(test_users)
run_naive_models(test_user=test_users, add_str="all_removed_repeat_limited_10000")     



# # TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/ngrams_matrix_train_age.json"
# with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
#     train_data = json.load(f)

# # RUN 3 : Second split 60/40 by age : every existing user
# TEST_PATH = MAIN_DIR / "results/predictions/train_test/test_age.csv"
# with open(TEST_PATH, 'r', encoding='utf-8') as f:
#     test_users = csv.reader(f, delimiter=";")
#     test_users = list(test_users)
# run_naive_models(test_user=test_users, add_str="all_AGE")   
    

# # # RUN 4 : Second split 60/40 by age : only movements 
# TEST_PATH = MAIN_DIR / "results/predictions/train_test/removed_repeat/test_age.csv"
# with open(TEST_PATH, 'r', encoding='utf-8') as f:
#     test_users = csv.reader(f, delimiter=";")
#     test_users = list(test_users)
# run_naive_models(test_user=test_users, add_str="all_removed_repeat_AGE")               


print(f"Everything done in {time.time()-t0:.2f}s")