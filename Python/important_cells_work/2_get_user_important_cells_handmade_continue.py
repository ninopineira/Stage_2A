# Gives the home/pseudo cell and activity (commute) cell of everyone based on the same principles that what there is
# in the raw dataset but with different morning and night range and a different way of classifying cells (based on their
# location and not on their IDs i.e. : "AAAAAA101" and "AAAAAA102" correspond to a unique position "AAAAAA")

# This script only processes data to get ready for plotting in the script barplot_user_important_cells.py

import csv
import tqdm
import pandas as pd
import re
from pathlib import Path
from typing import Callable
from collections import Counter
from utils import get_day



MAIN_DIR = Path(__file__).parent.parent.parent
dataset_name = Path(__file__).parent.name
INPUT_DIR = MAIN_DIR / f"Database/no_duplicate"
files = [file for file in INPUT_DIR.glob("*.csv")]
INTERMEDIATE_RESULT = MAIN_DIR / f"results/intermediate_result"
INTERMEDIATE_RESULT.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = MAIN_DIR / f"results/{dataset_name}/plots/important_cells_work"


MORNING_EVENING_PERIOD = {
    "Home_04h10-19h50_Activity_05h00-19h00": (15000, 71400), # Original time period for morning and evening
    
    "Home_04h00-19h00_Activity_05h00-18h00": (14400, 68400),
    "Home_04h00-19h00_Activity_06h00-18h00": (14400, 68400),
    "Home_04h00-20h00_Activity_05h00-19h00": (14400, 72000),
    "Home_04h00-20h00_Activity_06h00-18h00": (14400, 72000),
    "Home_04h00-20h00_Activity_06h00-19h00": (14400, 72000),
    
    "Home_05h00-19h00_Activity_06h00-18h00": (18000, 68400),
    "Home_05h00-20h00_Activity_06h00-18h00": (18000, 72000),
    "Home_05h00-20h00_Activity_06h00-19h00": (18000, 72000),
}
ACTIVITY_PERIOD = { # There is a small difference between the end of the morning and start of activity and same for the evening
    "Home_04h10-19h50_Activity_05h00-19h00": (18000, 68400), # Original working period (from 05h00 to 19h00)
    
    "Home_04h00-19h00_Activity_05h00-18h00": (18000, 64800),
    "Home_04h00-19h00_Activity_06h00-18h00": (21600, 64800),
    "Home_04h00-20h00_Activity_05h00-19h00": (18000, 68400),
    "Home_04h00-20h00_Activity_06h00-18h00": (21600, 64800),
    "Home_04h00-20h00_Activity_06h00-19h00": (21600, 68400),
    
    "Home_05h00-19h00_Activity_06h00-18h00": (21600, 64800),
    "Home_05h00-20h00_Activity_06h00-18h00": (21600, 64800),
    "Home_05h00-20h00_Activity_06h00-19h00": (21600, 68400),
}
USER_PRESENCE_CLASSIFICATION = {
    1 : "00h-24h",
    2 : "00h-yh",
    3 : "xh-24h",
    4 : "xh-yh"
}


def separate_day(cells : list[str], stamps : list[int], 
                morning : int | None = 15000, evening : int | None = 71400, 
                activity_start : int | None = 18000, activity_end : int | None = 68400,
                ):
    """
    Separates the cells and stamps list depending on the timestamps of the cells.
    The computation time is in O(n) as it only go though all index one time.
    I verified by hand that the function works as intended.
    
    Empty lists are returned for the period with no timestamps corresponding to the given period
    """
    
    n_records = len(stamps)
    cells_morning, cells_activity, cells_evening, stamps_activity = [],[],[],[]
    
    # Looking for the last timestamp at the end of the morning
    index_end_morning = 0
    while index_end_morning < n_records and stamps[index_end_morning] <= morning: 
        index_end_morning += 1
    
    # Looking for the start of activity time as the "Working hours" period can be different 
    # of the morning and evening threshold as specified in the document
    index_start_activity = index_end_morning
    while index_start_activity < n_records and stamps[index_start_activity] < activity_start: 
        index_start_activity += 1
    
    
    # Looking for the last timestamp of activity time
    index_end_activity = index_start_activity
    while index_end_activity < n_records and stamps[index_end_activity] <= activity_end:
        index_end_activity += 1


    # Looking for the start of evening time for the same reason as index_start_activity
    index_start_evening = index_end_activity 
    while index_start_evening < n_records and stamps[index_start_evening] < evening:
        index_start_evening += 1
    
    cells_morning = cells[:index_end_morning]
    cells_activity, stamps_activity = cells[index_start_activity:index_end_activity], stamps[index_start_activity:index_end_activity]
    cells_evening = cells[index_start_evening:]

    # User classification of its presence in the recorded zone
    # 1 : Here before morning threshold and here after evening threshold
    # 2 : Here before morning threshold and leaves before evening threshold
    # 3 : Comes after morning threshold and here after evening threshold
    # 4 : Comes after morning threshold and leaves before evening threshold
    
    if len(cells_morning) == 0:
        if len(cells_evening) == 0 : user_presence_classification = 4
        else : user_presence_classification = 3
    else:
        if len(cells_evening) > 0: user_presence_classification = 1
        else: user_presence_classification = 2
    
    return cells_morning, cells_activity, cells_evening, \
        stamps_activity, user_presence_classification

def get_cell_code(cell: str) -> str:
    """Extract base station if merge=True."""
    if cell == '': return cell
    match = re.match(r"([a-zA-Z]+)", cell)
    return match.group(1)

def get_cell_code2(cell: str) -> str:
    """Extract base station if merge=True."""
    if cell == '': return cell
    match = re.match(r"([a-zA-Z]+)", cell)
    return match.group(1)[1:]

MERGE = {
    "simple" : get_cell_code,
    "2g3g" : get_cell_code2
}

def get_base_stations_list(cells : list[str], merge_func : Callable[[str], str] | None = get_cell_code) -> list[str]:
    return [merge_func(c) for c in cells]

def get_base_stations_set(cells : set[str], merge_func : Callable[[str], str] | None = get_cell_code) -> set[str]:
    return {merge_func(c) for c in cells}

def get_user_cells(cells_morning, cells_activity, cells_evening, stamps_activity,
                   useful_merge_count_home, useful_merge_count_300, 
                   activity_stay_time : int | None = 18000, merge_func : Callable[[str], str] | None = get_cell_code):
    """
    Returns the home and activity cell of the user or None if these cells don't exist for the user ans the number of activity cells (cells in which the user stayed more than 300 minutes in continued time) and the reason why the cell is None if it is the case.

    A first pass is done to look for these cells depending on the cellid, if not successful a second pass is made while merging
    cells into base station. A global counter is used throughout all the users to see how impactful this second pass is.
    """
    
    # ======= #
    # Phase 0 # Init variables
    # ======= #
    Cell_Home, Cell_Activity = None, None
    reason_None = []
    merged_Cell_Home, merged_Cell_Activity = None, None
    Home_and_Activity_are_same = False
    merged_Home_and_Activity_are_same = False

    nb_activity_cells = 0
    
    n_morning = len(cells_morning)
    n_activity = len(cells_activity)
    n_evening = len(cells_evening)
    if n_morning == 0: 
        Cell_Home = ""
        merged_Cell_Home = ""
        reason_None.append("No_morning")
    if n_evening == 0:
        Cell_Home = ""
        merged_Cell_Home = ""
        reason_None.append("No_evening")
    if n_activity < 2:
        Cell_Activity = ""
        merged_Cell_Activity = ""
        reason_None.append("Not_enough_records")
    
    # ====== #
    # Exit 1 # if not enough records
    # ====== # 
    if Cell_Home == "" and Cell_Activity == "":
        return Cell_Home, Cell_Activity, Home_and_Activity_are_same, \
            merged_Cell_Home, merged_Cell_Activity, merged_Home_and_Activity_are_same, \
            reason_None, useful_merge_count_home, useful_merge_count_300, nb_activity_cells
    
    # ======= #
    # Phase 1 # First search of cells
    # ======= #
    set_cells_morning = set(cells_morning)
    set_cells_evening = set(cells_evening)
    
    # First search for a Cell_Home
    candidates_Home = set_cells_morning.intersection(set_cells_evening)

    if Cell_Home is None:
        if len(candidates_Home) != 0:
            for c in cells_morning:
                if c in candidates_Home:
                    Cell_Home = c # As specified, if we find such a cell present in both time period, we take the first occurence as our Cell_Home
                    break
        else: reason_None.append("no_candidates")
    
    # First search for activity cell
    # As precised by the tutor, the activity cell can be the same as the home cell in opposition of what is established in the documentation
    if Cell_Activity is None:
        # Check for cells having 300+ minutes of stay time (18000s)
        time_stayed_in_cells = Counter()
        old_cell = cells_activity[0]
        old_stamp = stamps_activity[0]
        for current_cell,current_timestamp in zip(cells_activity[1:], stamps_activity[1:]):
            if current_cell != old_cell:
                if current_timestamp-old_stamp < 4*3600: # if the time between two records is more than 4 hours, we consider that the user has disconnected and reconnected and we reset the time stayed in cells
                    time_stayed_in_cells[old_cell] += current_timestamp-old_stamp
                    if time_stayed_in_cells[old_cell] >= activity_stay_time: # if the user stayed enough time in the cell, we consider it as a candidate for Cell_Activity
                        old_cell = current_cell
                        old_stamp = current_timestamp

                    else: # if the user didn't stay enough time in the cell, we just update the time stayed in this cell and we don't consider it as a candidate for Cell_Activity
                        time_stayed_in_cells[old_cell] = 0
                        old_cell = current_cell
                        old_stamp = current_timestamp
                    
                else: # if the user is still in the same cell, we just update the time stayed in this cell
                    time_stayed_in_cells[old_cell] += 4*3600 # we consider that the user stayed 4 hours in the cell before disconnecting and reconnecting
                    if time_stayed_in_cells[old_cell] >= activity_stay_time: # if the user stayed enough time in the cell, we consider it as a candidate for Cell_Activity
                        old_cell = current_cell
                        old_stamp = current_timestamp

                    else: # if the user didn't stay enough time in the cell, we just update the time stayed in this cell and we don't consider it as a candidate for Cell_Activity
                        time_stayed_in_cells[old_cell] = 0
                        old_cell = current_cell
                        old_stamp = current_timestamp
            
            else: # if the user is still in the same cell, we just update the time stayed in this cell
                time_stayed_in_cells[old_cell] += current_timestamp-old_stamp
                old_cell = current_cell
                old_stamp = current_timestamp
        
        for cell in time_stayed_in_cells:
            if time_stayed_in_cells[cell] >= activity_stay_time:
                nb_activity_cells += 1

        candidates_activity = {cellid for cellid in time_stayed_in_cells.keys() if time_stayed_in_cells[cellid] >= activity_stay_time}

        for cell in cells_activity:
            if cell in candidates_activity:
                Cell_Activity = cell # first cell found is the Cell_Activity
                break
        
        if Cell_Activity is None:
            reason_None.append("Not_stayed_enough_in_cells")
    
    # ====== #
    # Exit 2 # If both cells are not None (either found cell or not enough stamps). Check if home and activity are same
    # ====== #
    if Cell_Home is not None and Cell_Activity is not None:
        merged_Cell_Home = ''
        merged_Cell_Activity = ''
        if Cell_Home != '' and Cell_Activity != '':
            if Cell_Home == Cell_Activity:
                Home_and_Activity_are_same = True
        return Cell_Home, Cell_Activity, Home_and_Activity_are_same, \
                merged_Cell_Home, merged_Cell_Activity, merged_Home_and_Activity_are_same, \
                reason_None, useful_merge_count_home, useful_merge_count_300, nb_activity_cells


    # ======= #
    # Phase 2 # Merge attempts by merging (e.g : BABCDE1 and UABCDE101 into ABCDE)
    # ======= #
    
    # Home
    if Cell_Home is None or Cell_Home != '':
        set_cells_morning = get_base_stations_set(set_cells_morning, merge_func=merge_func)
        set_cells_evening = get_base_stations_set(set_cells_evening, merge_func=merge_func)
        candidates_Home = set_cells_morning.intersection(set_cells_evening)
        if len(candidates_Home) != 0:
            cells_morning = get_base_stations_list(cells_morning, merge_func=merge_func)
            for c in cells_morning:
                if c in candidates_Home:
                    merged_Cell_Home = c # As specified, if we find such a cell present in both time period, we take the first occurence as our Cell_Home
                    if Cell_Home is None:
                        useful_merge_count_home += 1
                    break
        else: reason_None.append("no_candidates")
    
    
    # Activity
    if Cell_Activity is None or Cell_Activity != '':
        cells_activity = get_base_stations_list(cells_activity, merge_func=merge_func)
        
        time_stayed_in_cells = Counter()
        old_cell = cells_activity[0]
        old_stamp = stamps_activity[0]
        for current_cell,current_timestamp in zip(cells_activity[1:], stamps_activity[1:]):
            if current_cell != old_cell:
                if current_timestamp-old_stamp < 4*3600: # if the time between two records is more than 4 hours, we consider that the user has disconnected and reconnected and we reset the time stayed in cells
                    time_stayed_in_cells[old_cell] += current_timestamp-old_stamp
                    if time_stayed_in_cells[old_cell] >= activity_stay_time: # if the user stayed enough time in the cell, we consider it as a candidate for Cell_Activity
                        old_cell = current_cell
                        old_stamp = current_timestamp

                    else: # if the user didn't stay enough time in the cell, we just update the time stayed in this cell and we don't consider it as a candidate for Cell_Activity
                        time_stayed_in_cells[old_cell] = 0
                        old_cell = current_cell
                        old_stamp = current_timestamp
                    
                else: # if the user is still in the same cell, we just update the time stayed in this cell
                    time_stayed_in_cells[old_cell] += 4*3600 # we consider that the user stayed 4 hours in the cell before disconnecting and reconnecting
                    if time_stayed_in_cells[old_cell] >= activity_stay_time: # if the user stayed enough time in the cell, we consider it as a candidate for Cell_Activity
                        old_cell = current_cell
                        old_stamp = current_timestamp

                    else: # if the user didn't stay enough time in the cell, we just update the time stayed in this cell and we don't consider it as a candidate for Cell_Activity
                        time_stayed_in_cells[old_cell] = 0
                        old_cell = current_cell
                        old_stamp = current_timestamp
            
            else: # if the user is still in the same cell, we just update the time stayed in this cell
                time_stayed_in_cells[old_cell] += current_timestamp-old_stamp
                old_cell = current_cell
                old_stamp = current_timestamp     

        for cell in time_stayed_in_cells:
            if time_stayed_in_cells[cell] >= activity_stay_time and nb_activity_cells == 0:
                nb_activity_cells += 1

        
        candidates_activity = {cellid for cellid in time_stayed_in_cells.keys() if time_stayed_in_cells[cellid] >= activity_stay_time}

        for cell in cells_activity:
            if cell in candidates_activity:
                merged_Cell_Activity = cell
                if Cell_Activity is None:
                    useful_merge_count_300 += 1
                if merged_Cell_Activity == merged_Cell_Home: merged_Home_and_Activity_are_same = True
                break
                
        if merged_Cell_Activity is None:
            reason_None.append("Not_stayed_enough_in_cells")        

    # ====== #
    # Exit 3 #
    # ====== #
    return Cell_Home, Cell_Activity, Home_and_Activity_are_same, \
            merged_Cell_Home, merged_Cell_Activity, merged_Home_and_Activity_are_same, \
            reason_None, useful_merge_count_home, useful_merge_count_300 , nb_activity_cells
    
def process_user_activity(cells : list[str], stamps : list[int], 
                          morning : int | None = 15000, evening : int | None = 71400, 
                          activity_start : int | None = 18000, activity_end : int | None = 68400,
                          useful_merge_count_home : int | None = 0, useful_merge_count_300 : int | None = 0,
                          merge_func : Callable[[str], str] | None = get_cell_code):
    """
    Determine for the given user Cell_Home and Cell_Activity
    """
    reason_None = []    
    n_records = len(cells)
    
    
    
    # Separation of the full record list based on the defined period
    cells_morning, cells_activity, cells_evening, \
        stamps_activity, user_presence_classification = \
        separate_day(cells=cells, stamps=stamps,
                    morning = morning, evening = evening,
                    activity_start = activity_start, activity_end = activity_end)
    
    # ====== #
    # Exit 0 # if there is only 1 record for the day, completely useless in the classification
    # ====== #
    if n_records == 1: 
        reason_None.append("1_record")
        return user_presence_classification, \
            None, None, False, \
            None, None, False, \
            reason_None, useful_merge_count_home, useful_merge_count_300, 0
    
    # Get the actual data we are looking for : the important cells of the user.
    Cell_Home, Cell_Activity, Home_and_Activity_are_same, \
    merged_Cell_Home, merged_Cell_Activity, merged_Home_and_Activity_are_same, \
    reason_None, useful_merge_count_home, useful_merge_count_300, nb_activity_cells = \
        get_user_cells(cells_morning, cells_activity, cells_evening, stamps_activity,
                    useful_merge_count_home, useful_merge_count_300, 
                    activity_stay_time=18000, merge_func = merge_func)
    
    return user_presence_classification, \
            Cell_Home, Cell_Activity, Home_and_Activity_are_same, \
            merged_Cell_Home, merged_Cell_Activity, merged_Home_and_Activity_are_same, \
            reason_None, useful_merge_count_home, useful_merge_count_300, nb_activity_cells

# =========== #
# MAIN SCRIPT # 2 types of merging are tested
# =========== #

for merge_name,merge_func in MERGE.items():
    OUTPUT = INTERMEDIATE_RESULT / f"classified_dataset_merge_{merge_name}.csv"
    
    useful_merge_count_home = 0
    useful_merge_count_300 = 0

    result_days = {"user_id" : [],
                    "day" : [],
                    "user_presence_classification" : [],
                    
                    "Cell_Home" : [],
                    "Cell_Activity" : [],
                    "Same_Home_Activity" : [],
                    
                    "merged_Cell_Home" : [],
                    "merged_Cell_Activity" : [],
                    "merged_Same_Home_Activity" : [],
                    
                    "reason_None" : [],
                    "period" : [],
                    "working_period" : [],
                    "nb_activity_cells" : []
                    }

    for file in tqdm.tqdm(files):
        day = get_day(file)
        with open(file, mode='r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=';')
            for line in reader:
                user_id = line[0]
                user_cells = [c for c in line[8::2] if c]
                user_stamps = [int(ts) for ts in line[9::2] if ts]

                # Different temporal window are considred
                for label, (morning, evening) in MORNING_EVENING_PERIOD.items():
                    activity_start,activity_end = ACTIVITY_PERIOD[label]
                    
                    user_presence_classification, \
                        Cell_Home, Cell_Activity, Home_and_Activity_are_same, \
                        merged_Cell_Home, merged_Cell_Activity, merged_Home_and_Activity_are_same, \
                        reason_None, useful_merge_count_home, useful_merge_count_300, nb_activity_cells = \
                        process_user_activity(cells = user_cells, stamps = user_stamps, morning = morning, evening = evening, 
                            activity_start = activity_start, activity_end = activity_end,
                            useful_merge_count_home=useful_merge_count_home, useful_merge_count_300=useful_merge_count_300,
                            merge_func = merge_func)
                
                
                    result_days["user_id"].append(user_id)
                    result_days["day"].append(day)
                    result_days["user_presence_classification"].append(user_presence_classification)
                    
                    result_days["Cell_Home"].append(Cell_Home)
                    result_days["Cell_Activity"].append(Cell_Activity)
                    result_days["Same_Home_Activity"].append(Home_and_Activity_are_same)
                    
                    result_days["merged_Cell_Home"].append(merged_Cell_Home)
                    result_days["merged_Cell_Activity"].append(merged_Cell_Activity)
                    result_days["merged_Same_Home_Activity"].append(merged_Home_and_Activity_are_same)
                    
                    result_days["reason_None"].append(reason_None)
                    result_days["period"].append(label) 
                    result_days["working_period"].append( (activity_start, activity_end) )
                    result_days["nb_activity_cells"].append(nb_activity_cells)

    print(f"Le merging pour Home a été utile pour {useful_merge_count_home} cas")
    print(f"Le merging pour Activity a été utile pour {useful_merge_count_300} cas")

    df = pd.DataFrame(result_days)
    df.to_csv(OUTPUT, sep=";", header=True, index=False, index_label=False)