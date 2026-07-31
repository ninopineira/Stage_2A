# Prediction are made with the VOMM predictor with a full pipeline computing the ngrams without using the user 
# from which the sequence has been extracted to make the prediction. (Including the sequence in the training set)


# Every prediction are done on test split that does not have duplicate cells in a row (we created this csv by removing the
# repetitions with the script 3_deduplicate_csv.py in the dataset_creation folder in the python scripts of cd_142)

# However for the train split, most of the time, the data used have also contexts with repetitions, which are useless as
# they aren't taken into account while computing the contexts.

# But I've also added a prediction "80/20 random, removed_repeat test AND train"
# Which computes the train transition matrix based on a deduplicated form of the train_random.csv the same way we use
# the deduplicated test .csv files. With the expectation to have better results.


from pathlib import Path
import json
import csv
from utils import Metrics
from models import VOMM_V5
import time

MAIN_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = MAIN_DIR / "Database/no_duplicate"
OUTPUT_ACC_DIR = MAIN_DIR / "results/predictions/simple_predictor/metrics/V1"
OUTPUT_ACC_DIR.mkdir(parents=True, exist_ok=True)

metric_man = Metrics()
# ========== #
# MODEL RUNS #
# ========== #
t0 = time.time()

def VOMM_boosted(test_user, 
                 model : VOMM_V5 = None,
                 home_cell_boost = None,
                 decay = None, alpha_profile = None, max_context_length = 5):
    acc_res = {}
    meta_res = {"right" : 0,
                "wrong" : 0}
    top_ks = [1,3,5,10]

    preds_list = []
    true_next_list = []    
    # Test users
    k = 0
    i = 0
    number_of_preds = 0
    total_users = len(test_user)
    while i < total_users and k < 5000:
        user = test_user[i]
        n_cells = int(user[7])
        if n_cells < max_context_length+1:
            i = i + 1
        else:
            cells = user[8::2]
            timestamps = [ts for ts in user[9::2]]
            
            # Home / Activity boost prep
            if home_cell_boost is not None:
                home_cell = user[5] if user[5] != '' else None
                activity_cell = user[6] if user[6] != '' else None
            else: 
                home_cell = None,
                activity_cell = None
            
            for context_start in range(n_cells-max_context_length):
                next_index = max_context_length+context_start
                context = tuple(cells[context_start:next_index])
                last = cells[next_index-1]
                true_next = cells[next_index]
                timestamp = int(timestamps[next_index-1])



                # user_profile = model.build_user_profile(cells[:next_index],decay=decay) if decay is not None else None
                preds = model.predict_next_v5(context, 
                                            timestamp=timestamp, 
                                            home_cell=home_cell, activity_cell=activity_cell,alpha_boost = home_cell_boost,
                                            user_profile=None, alpha_profile = alpha_profile)
                true_next_list.append(true_next)
                preds_list.append(preds)
                
                if preds[0][0] == last:
                        if last == true_next: 
                            meta_res["right"] += 1
                        else : meta_res["wrong"] += 1
                number_of_preds += 1
                
            i += 1
            k += 1

    print(f"Successfully predicted {k} users\nTotal user seen {i} \nNumber of users skipped {i-k}")
    for top in top_ks:
        acc_res[f"ACC@{top}"] = metric_man.top_k_accuracy(preds_list=preds_list , true_next=true_next_list, k = top)
        acc_res[f"MAP@{top}"] = metric_man.map_k(preds_list=preds_list, true_next=true_next_list, k = top)
    acc_res["log-l_mean"] = metric_man.log_likelihood(preds_list=preds_list, true_next_list=true_next_list)

    meta_res["same_as_last_context_right"] = meta_res["right"] / number_of_preds
    meta_res["same_as_last_context_wrong"] = meta_res["wrong"] / number_of_preds
    meta_res["same_as_last_context_tot"] = meta_res["same_as_last_context_right"] + meta_res["same_as_last_context_wrong"]


    return acc_res, meta_res
    
# ===============================================================================
# PREDICTIONS 
# ===============================================================================
def predict(train_data_path, test_data_path,
            context_totals_train_data_path, 
            unique_train_data_path,
            unigram_train_data_path,
            discounts = [0.9], max_context_length = 5,
            output_metric_path : Path = None):
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_users = csv.reader(f, delimiter=";")
        test_users = list(test_users)
        
        
    # home_cell_boost = 0.3 # Need more test
    # profile_decay = [0.95] # Need more test
    # alphas_profile = [k/10 for k in range(1,10)] # Need more test
        
    res_metrics = {"meta" : {}}
        
    for discount in discounts:
        VOMM_model = VOMM_V5(train_data=train_data, max_order=max_context_length, 
                                    discount=discount,
                            context_totals_train_data_path = context_totals_train_data_path,
                            unique_train_data_path = unique_train_data_path,
                            unigram_train_data_path = unigram_train_data_path,
                            DATASET_DIR = DATASET_DIR)

        model_name = f"rdm_nrepeat_{discount}"
        res_metrics[model_name], res_metrics["meta"][model_name] = VOMM_boosted(test_user=test_users, model = VOMM_model)
    
    
    with open(output_metric_path, mode="w") as f:
        json.dump(res_metrics, f, indent=2)

    print(f"Everything done in {time.time()-t0:.2f}s")

# ======= # ===============================================================
# == 1 == # ===============================================================
# ======= # ===============================================================

# RANDOM 80/20 - TEST REMOVED REPEAT
TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/ngrams_matrix_train_random.json"
TEST_PATH = MAIN_DIR / "results/predictions/train_test/removed_repeat/test_random.csv"
context_totals_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/context_totals_train.json"
unique_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/unique_followers_train.json"
unigram_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/unigram_train.json"

predict(train_data_path = TRAIN_DATA_PATH, test_data_path = TEST_PATH,
        context_totals_train_data_path = context_totals_train_data_path,
        unique_train_data_path = unique_train_data_path,
        unigram_train_data_path = unigram_train_data_path,
        discounts = [0.5, 0.75, 0.9], max_context_length= 5,
        output_metric_path = OUTPUT_ACC_DIR / "V1_discount_study.json")


# RANDOM 80/20 merged - TEST REMOVED REPEAT
# TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/merge/2g3g_matrix_train_random.json"
# TEST_PATH = MAIN_DIR / "results/predictions/train_test/merge/test_random_removed_repeat.csv"
# context_totals_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/merge/context_totals_train.json"
# unique_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/merge/unique_followers_train.json"
# unigram_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/merge/unigram_train.json"

# predict(train_data_path = TRAIN_DATA_PATH, test_data_path = TEST_PATH,
#         context_totals_train_data_path = context_totals_train_data_path,
#         unique_train_data_path = unique_train_data_path,
#         unigram_train_data_path = unigram_train_data_path,
#         discounts = [0.9], max_context_length= 5,
#         output_metric_path = OUTPUT_ACC_DIR / "V1_random_no_repeat_merge2g3g.json")


# ======= # ===============================================================
# == 2 == # ===============================================================
# ======= # ===============================================================

# 80/20 random, removed_repeat test AND train
# TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/removed_repeat/ngrams_matrix_train_random_removed_repeat.json"
# TEST_PATH = MAIN_DIR / "results/predictions/train_test/removed_repeat/test_random.csv"
# context_totals_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/removed_repeat/context_totals_train.json"
# unique_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/removed_repeat/unique_followers_train.json"
# unigram_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/removed_repeat/unigram_train.json"

# predict(train_data_path = TRAIN_DATA_PATH, test_data_path = TEST_PATH,
#         context_totals_train_data_path = context_totals_train_data_path,
#         unique_train_data_path = unique_train_data_path,
#         unigram_train_data_path = unigram_train_data_path,
#         discounts = [0.9], max_context_length= 5,
#         output_metric_path = OUTPUT_ACC_DIR / "V1_random_no_repeat_train_and_test.json")

# RANDOM 80/20 merged removed_repeat test AND train
# TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/merge/2g3g_matrix_train_random_removed_repeat.json"
# TEST_PATH = MAIN_DIR / "results/predictions/train_test/merge/test_random_removed_repeat.csv"
# context_totals_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/merge/context_totals_train_removed_repeat.json"
# unique_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/merge/unique_followers_train_removed_repeat.json"
# unigram_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/merge/unigram_train_removed_repeat.json"

# predict(train_data_path = TRAIN_DATA_PATH, test_data_path = TEST_PATH,
#         context_totals_train_data_path = context_totals_train_data_path,
#         unique_train_data_path = unique_train_data_path,
#         unigram_train_data_path = unigram_train_data_path,
#         discounts = [0.9], max_context_length= 5,
#         output_metric_path = OUTPUT_ACC_DIR / "V1_random_no_repeat_test_and_train_merge2g3g.json")


# ======= # ===============================================================
# == 3 == # ===============================================================
# ======= # ===============================================================

# RANDOM 80/20 - WITH REPEAT
# TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/ngrams_matrix_train_random.json"
# TEST_PATH = MAIN_DIR / "results/predictions/train_test/test_random.csv"
# context_totals_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/ngrams/context_totals_train.json"
# unique_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/ngrams/unique_followers_train.json"
# unigram_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/ngrams/unigram_train.json"

# predict(train_data_path = TRAIN_DATA_PATH, test_data_path = TEST_PATH,
#         context_totals_train_data_path = context_totals_train_data_path,
#         unique_train_data_path = unique_train_data_path,
#         unigram_train_data_path = unigram_train_data_path,
#         discounts = [0.9], max_context_length= 5,
#         output_metric_path = OUTPUT_ACC_DIR / "V1_random.json")



# # RANDOM 80/20 merged - WITH REPEAT
# TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/merge/2g3g_matrix_train_random.json"
# TEST_PATH = MAIN_DIR / "results/predictions/train_test/merge/test_random.csv"
# context_totals_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/merge/context_totals_train.json"
# unique_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/merge/unique_followers_train.json"
# unigram_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/merge/unigram_train.json"

# predict(train_data_path = TRAIN_DATA_PATH, test_data_path = TEST_PATH,
#         context_totals_train_data_path = context_totals_train_data_path,
#         unique_train_data_path = unique_train_data_path,
#         unigram_train_data_path = unigram_train_data_path,
#         discounts = [0.9], max_context_length= 5,
#         output_metric_path = OUTPUT_ACC_DIR / "V1_random_merge2g3g.json")


# ======= # ===============================================================
# == 4 == # ===============================================================
# ======= # ===============================================================

# Class 1
# TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/class/matrix_train_class1.json"
# TEST_PATH = MAIN_DIR / "results/predictions/train_test/class/class1_test_removed_repeat.csv"
# context_totals_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/class/context_totals_train_class1.json"
# unique_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/class/unique_followers_train_class1.json"
# unigram_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/class/unigram_train_class1.json"

# predict(train_data_path = TRAIN_DATA_PATH, test_data_path = TEST_PATH,
#         context_totals_train_data_path = context_totals_train_data_path,
#         unique_train_data_path = unique_train_data_path,
#         unigram_train_data_path = unigram_train_data_path,
#         discounts = [0.9], max_context_length= 5,
#         output_metric_path = OUTPUT_ACC_DIR / "V1_class1_metrics.json")

# # Weekdays
# TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/week/matrix_train_weekdays.json"
# TEST_PATH = MAIN_DIR / "results/predictions/train_test/week/weekdays_test_removed_repeat.csv"
# context_totals_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/week/context_totals_train_weekdays.json"
# unique_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/week/unique_followers_train_weekdays.json"
# unigram_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/week/unigram_train_weekdays.json"

# predict(train_data_path = TRAIN_DATA_PATH, test_data_path = TEST_PATH,
#         context_totals_train_data_path = context_totals_train_data_path,
#         unique_train_data_path = unique_train_data_path,
#         unigram_train_data_path = unigram_train_data_path,
#         discounts = [0.9], max_context_length= 5,
#         output_metric_path = OUTPUT_ACC_DIR / "V1_weekdays_metrics.json")

# # Weekend
# TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/week/matrix_train_weekend.json"
# TEST_PATH = MAIN_DIR / "results/predictions/train_test/week/weekend_test_removed_repeat.csv"
# context_totals_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/week/context_totals_train_weekend.json"
# unique_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/week/unique_followers_train_weekend.json"
# unigram_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/week/unigram_train_weekend.json"

# predict(train_data_path = TRAIN_DATA_PATH, test_data_path = TEST_PATH,
#         context_totals_train_data_path = context_totals_train_data_path,
#         unique_train_data_path = unique_train_data_path,
#         unigram_train_data_path = unigram_train_data_path,
#         discounts = [0.9], max_context_length= 5,
#         output_metric_path = OUTPUT_ACC_DIR / "V1_weekend_metrics.json")

# # Class1 + Weekdays
# TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/class_week/matrix_train_class1_weekdays.json"
# TEST_PATH = MAIN_DIR / "results/predictions/train_test/class_week/class1_weekdays_test_removed_repeat.csv"
# context_totals_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/class_week/context_totals_train_class1_weekdays.json"
# unique_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/class_week/unique_followers_train_class1_weekdays.json"
# unigram_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/class_week/unigram_train_class1_weekdays.json"

# predict(train_data_path = TRAIN_DATA_PATH, test_data_path = TEST_PATH,
#         context_totals_train_data_path = context_totals_train_data_path,
#         unique_train_data_path = unique_train_data_path,
#         unigram_train_data_path = unigram_train_data_path,
#         discounts = [0.9], max_context_length= 5,
#         output_metric_path = OUTPUT_ACC_DIR / "V1_class1_weekdays_metrics.json")

# # Class1 + Weekend
# TRAIN_DATA_PATH = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/class_week/matrix_train_class1_weekend.json"
# TEST_PATH = MAIN_DIR / "results/predictions/train_test/class_week/class1_weekend_test_removed_repeat.csv"
# context_totals_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/class_week/context_totals_train_class1_weekend.json"
# unique_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/class_week/unique_followers_train_class1_weekend.json"
# unigram_train_data_path = MAIN_DIR / "results/predictions/simple_predictor/transition_matrix/training_matrix/class_week/unigram_train_class1_weekend.json"
# predict(train_data_path = TRAIN_DATA_PATH, test_data_path = TEST_PATH,
#         context_totals_train_data_path = context_totals_train_data_path,
#         unique_train_data_path = unique_train_data_path,
#         unigram_train_data_path = unigram_train_data_path,
#         discounts = [0.9], max_context_length= 5,
#         output_metric_path = OUTPUT_ACC_DIR / "V1_class1_weekend_metrics.json")