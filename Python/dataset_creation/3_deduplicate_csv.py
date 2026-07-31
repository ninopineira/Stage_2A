import csv
from pathlib import Path

def deduplicate(test_path, output_path):
    new_csv=[]

    with open(test_path, mode="r", newline='') as f:
        reader = csv.reader(f,delimiter=";")
        for user in reader:
            records = user[8:]
            deduplicated_records = records[:2]
            for i in range(2,len(records),2):
                if records[i-2] == records[i]:
                    continue
                else:
                    deduplicated_records.append(records[i])
                    deduplicated_records.append(records[i+1])
            new_csv.append(user[:7] + [len(deduplicated_records)//2] + deduplicated_records)
    with open(output_path, mode="w", newline='') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(new_csv)

MAIN_DIR = Path(__file__).parent.parent.parent

# TEST_SPLIT_PATH = MAIN_DIR / "results/predictions/train_test/test_random.csv"
# OUTPUT_PATH = MAIN_DIR / f"results/predictions/train_test/removed_repeat/{TEST_SPLIT_PATH.stem}.csv"
# OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
# deduplicate(TEST_SPLIT_PATH,OUTPUT_PATH)

# ====================================#
# TRAIN RANDOM AND TRAIN RANDOM MERGE #
# ====================================#
# TRAIN_SPLIT_PATH = MAIN_DIR / "results/predictions/train_test/train_random.csv"
# OUTPUT_PATH = MAIN_DIR / f"results/predictions/train_test/removed_repeat/{TRAIN_SPLIT_PATH.stem}_removed_repeat.csv"
# deduplicate(TRAIN_SPLIT_PATH,OUTPUT_PATH)

# Pipeline A: deduplicate the test split (consumed by the prediction pipelines)
TEST_SPLIT_PATH = MAIN_DIR / "results/predictions/train_test/test_random.csv"
OUTPUT_PATH = MAIN_DIR / f"results/predictions/train_test/removed_repeat/{TEST_SPLIT_PATH.stem}.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
deduplicate(TEST_SPLIT_PATH,OUTPUT_PATH)


# TEST_SPLIT_PATH2 = MAIN_DIR / "results/predictions/train_test/test_age.csv"
# OUTPUT_PATH2 = MAIN_DIR / f"results/predictions/train_test/removed_repeat/{TEST_SPLIT_PATH2.stem}.csv"
# deduplicate(TEST_SPLIT_PATH2,OUTPUT_PATH2)


# Class 1
# TEST_SPLIT_PATH = MAIN_DIR / "results/predictions/train_test/class/class1_test.csv"
# OUTPUT_PATH = MAIN_DIR / f"results/predictions/train_test/class/{TEST_SPLIT_PATH.stem}_removed_repeat.csv"
# deduplicate(TEST_SPLIT_PATH,OUTPUT_PATH)

# # Weekdays
# TEST_SPLIT_PATH = MAIN_DIR / "results/predictions/train_test/week/weekdays_test.csv"
# OUTPUT_PATH = MAIN_DIR / f"results/predictions/train_test/week/{TEST_SPLIT_PATH.stem}_removed_repeat.csv"
# deduplicate(TEST_SPLIT_PATH,OUTPUT_PATH)


# # Weekend
# TEST_SPLIT_PATH = MAIN_DIR / "results/predictions/train_test/week/weekend_test.csv"
# OUTPUT_PATH = MAIN_DIR / f"results/predictions/train_test/week/{TEST_SPLIT_PATH.stem}_removed_repeat.csv"
# deduplicate(TEST_SPLIT_PATH,OUTPUT_PATH)


# # Class 1 + weekdays
# TEST_SPLIT_PATH = MAIN_DIR / "results/predictions/train_test/class_week/class1_weekdays_test.csv"
# OUTPUT_PATH = MAIN_DIR / f"results/predictions/train_test/class_week/{TEST_SPLIT_PATH.stem}_removed_repeat.csv"
# deduplicate(TEST_SPLIT_PATH,OUTPUT_PATH)

# # Class 1 + weekend
# TEST_SPLIT_PATH = MAIN_DIR / "results/predictions/train_test/class_week/class1_weekend_test.csv"
# OUTPUT_PATH = MAIN_DIR / f"results/predictions/train_test/class_week/{TEST_SPLIT_PATH.stem}_removed_repeat.csv"
# deduplicate(TEST_SPLIT_PATH,OUTPUT_PATH)


# 2g3g merge
# TEST_SPLIT_PATH = MAIN_DIR / "results/predictions/train_test/merge/test_random.csv"
# OUTPUT_PATH = MAIN_DIR / f"results/predictions/train_test/merge/{TEST_SPLIT_PATH.stem}_removed_repeat.csv"
# deduplicate(TEST_SPLIT_PATH,OUTPUT_PATH)




