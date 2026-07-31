import csv
import tqdm
import json
import random
from pathlib import Path
from collections import Counter

from utils import save_csv

MAIN_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = MAIN_DIR / "Database/no_duplicate"
OUTPUT_DIR = MAIN_DIR / "results/predictions/train_test"


files = [path for path in DATASET_DIR.glob("*.csv")]

SEED = 67
random.seed(SEED)

def save_csv(data,path : Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(data)

train_data_complete = []
test_data_complete  = []

# ================= #
# FULL RANDOM SPLIT #
# ================= #
for file in tqdm.tqdm(files):
    with open(file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=";")
        csv_data = list(reader)
        
    # Random shuffle of data + split 80% Train / 20% Test
    random.shuffle(csv_data)
    n = len(csv_data)
    train_test_sep = int(n*0.8)
    train_data = csv_data[:train_test_sep]
    test_data = csv_data[train_test_sep:]

    train_data_complete += train_data
    test_data_complete += test_data

output_train = OUTPUT_DIR / "train_random.csv"
output_test  = OUTPUT_DIR / "test_random.csv"
output_train.parent.mkdir(parents=True, exist_ok=True)
save_csv(data=train_data_complete, path=output_train)
save_csv(data=test_data_complete , path=output_test)


# =========================== #
# FULL RANDOM SPLIT ON MERGED #
# =========================== #

# files = [MAIN_DIR / "Database/merge/2g3g_merge.csv"]

# for file in files:
#     with open(file, 'r', encoding='utf-8', newline='') as f:
#         reader = csv.reader(f, delimiter=";")
#         csv_data = list(reader)
        
#     # Random shuffle of data + split 60% Train / 20% Validation / 20% Test
#     random.shuffle(csv_data)
#     n = len(csv_data)
#     train_test_sep = int(n*0.8)
#     train_data = csv_data[:train_test_sep]
#     test_data = csv_data[train_test_sep:]

#     train_data_complete += train_data
#     test_data_complete += test_data

# output_train = OUTPUT_DIR / "merge/train_random.csv"
# output_test  = OUTPUT_DIR / "merge/test_random.csv"
# output_train.parent.mkdir(parents=True, exist_ok=True)
# save_csv(data=train_data_complete, path=output_train)
# save_csv(data=test_data_complete , path=output_test)


# ========================================== #
# FULL RANDOM SPLIT ON MERGED + CLASS 1 ONLY # (for movement prediction)
# ========================================== #

# file = MAIN_DIR / "Database/class_merge/class1_2g3g_merge.csv"

# with open(file, 'r', encoding='utf-8', newline='') as f:
#     reader = csv.reader(f, delimiter=";")
#     csv_data = list(reader)
        
# # Random shuffle of data + split 80% Train / 20% Test
# random.shuffle(csv_data)
# n = len(csv_data)
# train_test_sep = int(n*0.8)
# train_data = csv_data[:train_test_sep]
# test_data = csv_data[train_test_sep:]

# train_data_complete += train_data
# test_data_complete += test_data

# output_train = OUTPUT_DIR / "class_merge/class1_train_random.csv"
# output_test  = OUTPUT_DIR / "class_merge/class1_test_random.csv"
# output_train.parent.mkdir(parents=True, exist_ok=True)
# save_csv(data=train_data_complete, path=output_train)
# save_csv(data=test_data_complete , path=output_test)













# # =================== #
# # SPLIT BY AGE OR NOT #
# # =================== #
# train_data_complete = []
# test_data_complete  = []
# counter = {}
# counter["total"] = Counter()

# for file in tqdm.tqdm(files):
#     filename = file.stem
#     counter[filename] = Counter()
#     with open(file, 'r', encoding='utf-8', newline='') as f:
#         reader = csv.reader(f, delimiter=";")
#         csv_data = list(reader)
        
#     for user in csv_data:
#         if user[1] == "":
#             train_data_complete.append(user)
#             counter[filename].update(["train"])
#         else:
#             test_data_complete.append(user)
#             counter[filename].update(["test"])
    
#     counter["total"]["train"] += counter[filename]["train"]
#     counter["total"]["test"] += counter[filename]["test"]

# output_train = OUTPUT_DIR / "train_age.csv" # Doesn't have age
# output_test  = OUTPUT_DIR / "test_age.csv"  # Have age
# output_train.parent.mkdir(parents=True, exist_ok=True)

# n = len(train_data_complete)

# save_csv(data=train_data_complete,path=output_train)
# save_csv(data=test_data_complete, path=output_test)


# with open(OUTPUT_DIR / "counter_age.json", mode='w') as f:
#     json.dump(counter,f)



# =================== #
# SPLIT BY CLASS      #
# =================== #
# DATASET_PATH = MAIN_DIR / f"Database/class/class1.csv"
# with open(DATASET_PATH, 'r', encoding='utf-8', newline='') as f:
#     reader = csv.reader(f, delimiter=";")
#     users = list(reader)
# n = len(users)
# split = int(n*0.8)
# train, test = users[:split], users[split:]
# output_train = OUTPUT_DIR / "class/class1_train.csv"
# output_test = OUTPUT_DIR / "class/class1_test.csv"
# save_csv(data=train,path=output_train)
# save_csv(data=test,path=output_test)


# # ================ #
# # SPLIT BY WEEKEND #
# # ================ #
# DATASET_PATH = MAIN_DIR / f"Database/week/weekdays.csv"
# with open(DATASET_PATH, 'r', encoding='utf-8', newline='') as f:
#     reader = csv.reader(f, delimiter=";")
#     users = list(reader)
# n = len(users)
# split = int(n*0.8)
# train, test = users[:split], users[split:]
# output_train = OUTPUT_DIR / "week/weekdays_train.csv"
# output_test = OUTPUT_DIR / "week/weekdays_test.csv"
# save_csv(data=train,path=output_train)
# save_csv(data=test,path=output_test)

# DATASET_PATH = MAIN_DIR / f"Database/week/weekend.csv"
# with open(DATASET_PATH, 'r', encoding='utf-8', newline='') as f:
#     reader = csv.reader(f, delimiter=";")
#     users = list(reader)
# n = len(users)
# split = int(n*0.8)
# train, test = users[:split], users[split:]
# output_train = OUTPUT_DIR / "week/weekend_train.csv"
# output_test = OUTPUT_DIR / "week/weekend_test.csv"
# save_csv(data=train,path=output_train)
# save_csv(data=test,path=output_test)

# # ========================== #
# # SPLIT BY CLASS AND WEEKEND #
# # ========================== #
# DATASET_PATH = MAIN_DIR / f"Database/class_week/class1_weekdays.csv"
# with open(DATASET_PATH, 'r', encoding='utf-8', newline='') as f:
#     reader = csv.reader(f, delimiter=";")
#     users = list(reader)
# n = len(users)
# split = int(n*0.8)
# train, test = users[:split], users[split:]
# output_train = OUTPUT_DIR / "class_week/class1_weekdays_train.csv"
# output_test = OUTPUT_DIR / "class_week/class1_weekdays_test.csv"
# save_csv(data=train,path=output_train)
# save_csv(data=test,path=output_test)


# DATASET_PATH = MAIN_DIR / f"Database/class_week/class1_weekend.csv"
# with open(DATASET_PATH, 'r', encoding='utf-8', newline='') as f:
#     reader = csv.reader(f, delimiter=";")
#     users = list(reader)
# n = len(users)
# split = int(n*0.8)
# train, test = users[:split], users[split:]
# output_train = OUTPUT_DIR / "class_week/class1_weekend_train.csv"
# output_test = OUTPUT_DIR / "class_week/class1_weekend_test.csv"
# save_csv(data=train,path=output_train)
# save_csv(data=test,path=output_test)