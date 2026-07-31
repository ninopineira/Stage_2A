# Recreate a full dataset into a single file (or 2 if we split the weekends) with only users belonging to a certain class



from pathlib import Path
import pandas as pd
import csv
from utils import is_weekend, get_day, save_csv
import tqdm

MAIN_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = MAIN_DIR / "Database/no_duplicate"
CLASSIFICATION_PATH = MAIN_DIR / "results/intermediate_result/cell_classification_2.csv"

def merge_cell_id(cell : str):
    if cell.startswith(('B','D')):
        return cell[1:-1]
    else:
        return cell[1:-3]


files = [path for path in DATASET_DIR.glob("*.csv")]
df = pd.read_csv(CLASSIFICATION_PATH, sep=";", index_col=['user_id'])

# sep weekend, weekdays
user_workdays = []
user_weekend  = []

# Class 1 only
always_here = []

# Class 1 + sep weekend, weekdays
always_here_users_workdays = []
always_here_users_weekend  = []



# 2G/3G
merged = []
class1_merged = []


for file in files:
    day = get_day(file)
    is_wkend = is_weekend(day)
    
    with open(file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=";")
        for user in tqdm.tqdm(reader):
            
            
            
            
            user_id = int(user[0])
            class_1 = (df.loc[user_id].user_presence_classification == 1)
            
            # if class_1:
            #     always_here.append(user)
            #     if is_wkend: 
            #         always_here_users_weekend.append(user)
            #         user_weekend.append(user)
            #     else: 
            #         always_here_users_workdays.append(user)
            #         user_workdays.append(user)
            # elif is_wkend:user_weekend.append(user)
            # else: user_workdays.append(user)

            # 2g/3g merge
            
            # if int(user[7]) > 5:
            #     old_records = user[8:]
            #     merged_records = [merge_cell_id(old_records[i]) if i%2 == 0 else old_records[i] for i in range(len(old_records)) ]
                
            #     merged.append(user[:8] + merged_records)
             
                
            # 2g/3g + class 1
            if class_1 and int(user[7]) <= 512:
                old_records = user[8:]
                merged_records = [merge_cell_id(old_records[i]) if i%2 == 0 else old_records[i] for i in range(len(old_records)) ]
                class1_merged.append(user[:8] + merged_records)
            


# output_class_weekdays   = MAIN_DIR / "Database/class_week/class1_weekdays.csv"
# output_class_weekend    = MAIN_DIR / "Database/class_week/class1_weekend.csv"

# output_class            = MAIN_DIR / "Database/class/class1.csv"

# output_weekedays        = MAIN_DIR / "Database/week/weekdays.csv"
# output_weekend          = MAIN_DIR / "Database/week/weekend.csv"

# output_merge            = MAIN_DIR / "Database/merge/2g3g_merge.csv"

output_merge_class      = MAIN_DIR / "Database/class_merge/class1_2g3g_merge.csv"

# save_csv(path = output_class,           data = always_here)
# save_csv(path = output_class_weekend,   data = always_here_users_weekend)
# save_csv(path = output_class_weekdays,  data = always_here_users_workdays)
# save_csv(path = output_weekedays,       data = user_workdays)
# save_csv(path = output_weekend,         data = user_weekend)
# save_csv(path = output_merge, data = merged)
save_csv(path = output_merge_class, data = class1_merged)