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
                activity_start : int | None = 18000, activity_end : int | None = 68400,
                ):
    """
    Separates the cells and stamps list depending on the timestamps of the cells.
    Also returns the last record before the window and the first record after,
    used to handle boundary continuity in activity cell detection.
    """

    n_records = len(stamps)

    index_start_activity = 0
    while index_start_activity < n_records and stamps[index_start_activity] < activity_start:
        index_start_activity += 1

    index_end_activity = index_start_activity
    while index_end_activity < n_records and stamps[index_end_activity] <= activity_end:
        index_end_activity += 1

    cells_activity  = cells[index_start_activity:index_end_activity]
    stamps_activity = stamps[index_start_activity:index_end_activity]

    pre_cell  = cells[index_start_activity - 1]  if index_start_activity > 0       else None
    pre_stamp = stamps[index_start_activity - 1] if index_start_activity > 0       else None
    post_cell  = cells[index_end_activity]        if index_end_activity < n_records else None
    post_stamp = stamps[index_end_activity]       if index_end_activity < n_records else None

    return cells_activity, stamps_activity, pre_cell, pre_stamp, post_cell, post_stamp

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
    "no_merge" : lambda x: x,
    "simple" : get_cell_code,
    "2g3g" : get_cell_code2
}

def get_base_stations_list(cells : list[str], merge_func : Callable[[str], str] | None = get_cell_code) -> list[str]:
    return [merge_func(c) for c in cells]

def get_base_stations_set(cells : set[str], merge_func : Callable[[str], str] | None = get_cell_code) -> set[str]:
    return {merge_func(c) for c in cells}

def get_home_cell(cells : list[str], stamps : list[int], activity_start : int = 18000, activity_end : int = 68400, merge_func : Callable[[str], str] = lambda x: x) -> str | None:
    """
    Returns the home cell of the user, or None if no home cell can be identified.

    Conditions (applied after deduplication of consecutive identical records):
      - The cell must appear at least once in the morning window [0:00, 04:10] (0–15000 s)
      - The cell must appear at least once in the evening window [19:50, 23:59:59] (71400–86399 s)

    The merge_func is applied to all cell ids before the search.
    The first qualifying cell encountered in the record sequence is returned.
    """
    MORNING_END   = 4 * 3600 + 10 * 60   # 04:10:00 = 15 000 s
    EVENING_START = 19 * 3600 + 50 * 60  # 19:50:00 = 71 400 s

    if len(cells) < 2:
        return None

    cells_m = [merge_func(c) for c in cells]
    

    morning_cells = {c for c, t in zip(cells_m, stamps) if t <= MORNING_END}
    evening_cells = {c for c, t in zip(cells_m, stamps) if t >= EVENING_START}

    candidates = morning_cells & evening_cells
    if not candidates:
        return None

    for c in cells_m:
        if c in candidates:
            return c

    return None



def get_user_cells_cont_no_gap(cells_activity, stamps_activity, useful_merge_count_300, activity_stay_time : int | None = 18000, merge_func : Callable[[str], str] = lambda x: x, cells_full=None, stamps_full=None):
    """
    Returns the home and activity cell of the user or None if these cells don't exist for the user ans the number of activity cells (cells in which the user stayed more than 300 minutes in continued time) and the reason why the cell is None if it is the case.

    A first pass is done to look for these cells depending on the cellid, if not successful a second pass is made while merging
    cells into base station. A global counter is used throughout all the users to see how impactful this second pass is.
    """

    # ======= #
    # Phase 0 # Init variables
    # ======= #
    Cell_Activity = None
    reason_None = []

    nb_activity_cells = []
    home_cell = get_home_cell(cells_full, stamps_full, merge_func=merge_func) if cells_full is not None else None
    
    n_activity = len(cells_activity)
    if n_activity < 2:
        Cell_Activity = ""
        reason_None.append("Not_enough_records")
    
    # ====== #
    # Exit 1 # if not enough records
    # ====== # 
    if Cell_Activity == "":
        return nb_activity_cells, reason_None, useful_merge_count_300, len(nb_activity_cells), [], []
    
    # ======= #
    # Phase 1 # First search of cells
    # ======= #
    
    # First search for activity cell


    if Cell_Activity is None:
        # Check for cells having 300+ minutes of stay time (18000s)
        time_stayed_in_cells = Counter()
        records_per_cell     = Counter(cells_activity)
        old_cell = cells_activity[0]
        old_stamp = stamps_activity[0]
        for current_cell,current_timestamp in zip(cells_activity[1:], stamps_activity[1:]):
            if current_cell != old_cell:
                if current_timestamp-old_stamp < 4*3600 + 600: # if the time between two records is more than 4 hours, we consider that the user has disconnected and reconnected and we reset the time stayed in cells
                    time_stayed_in_cells[old_cell] += current_timestamp-old_stamp
                    old_cell = current_cell
                    old_stamp = current_timestamp
                    
                else:
                    time_stayed_in_cells[old_cell] = 0
                    old_cell = current_cell
                    old_stamp = current_timestamp
                    
                if time_stayed_in_cells[old_cell] < activity_stay_time:
                    time_stayed_in_cells[old_cell] = 0
            
            else: # if the user is still in the same cell, we just update the time stayed in this cell
                if current_timestamp-old_stamp < 4*3600 + 600: # if the time between two records is more than 4 hours, we consider that the user has disconnected and reconnected and we reset the time stayed in cells
                    time_stayed_in_cells[old_cell] += current_timestamp-old_stamp
                    old_cell = current_cell
                    old_stamp = current_timestamp
                    
                else:
                    if time_stayed_in_cells[old_cell] < activity_stay_time:
                        time_stayed_in_cells[old_cell] = 0
                        old_cell = current_cell
                        old_stamp = current_timestamp
        
        for cell in time_stayed_in_cells:
            if time_stayed_in_cells[cell] >= activity_stay_time and cell not in nb_activity_cells and cell != home_cell:
                nb_activity_cells.append(cell)
        nb_activity_cells.sort(key=lambda c: cells_activity.count(c), reverse=True)
        cell_times   = [time_stayed_in_cells[c] for c in nb_activity_cells]
        cell_records = [records_per_cell[c]      for c in nb_activity_cells]


        candidates_activity = {cellid for cellid in time_stayed_in_cells.keys() if time_stayed_in_cells[cellid] >= activity_stay_time and cellid != home_cell}

        for cell in cells_activity:
            if cell in candidates_activity:
                Cell_Activity = cell # first cell found is the Cell_Activity
                break

        if Cell_Activity is None:
            reason_None.append("Not_stayed_enough_in_cells")

    # ====== #
    # Exit 3 #
    # ====== #
    return nb_activity_cells, reason_None, useful_merge_count_300, len(nb_activity_cells), cell_times, cell_records

def get_user_cells_cont_gap(cells_activity, stamps_activity, useful_merge_count_300, activity_stay_time : int | None = 18000, merge_func : Callable[[str], str] = lambda x: x, cells_full=None, stamps_full=None):
    """
    Returns the home and activity cell of the user or None if these cells don't exist for the user ans the number of activity cells (cells in which the user stayed more than 300 minutes in continued time) and the reason why the cell is None if it is the case.

    A first pass is done to look for these cells depending on the cellid, if not successful a second pass is made while merging
    cells into base station. A global counter is used throughout all the users to see how impactful this second pass is.
    """

    # ======= #
    # Phase 0 # Init variables
    # ======= #
    Cell_Activity = None
    reason_None = []

    nb_activity_cells = []
    home_cell = get_home_cell(cells_full, stamps_full, merge_func=merge_func) if cells_full is not None else None
    
    n_activity = len(cells_activity)
    if n_activity < 2:
        Cell_Activity = ""
        reason_None.append("Not_enough_records")
    
    # ====== #
    # Exit 1 # if not enough records
    # ====== # 
    if Cell_Activity == "":
        return nb_activity_cells, reason_None, useful_merge_count_300, len(nb_activity_cells), [], []
    
    # ======= #
    # Phase 1 # First search of cells
    # ======= #
    
    # First search for activity cell

    if Cell_Activity is None:
        # Check for cells having 300+ minutes of stay time (18000s)
        time_stayed_in_cells = Counter()
        records_per_cell     = Counter(cells_activity)
        old_cell = cells_activity[0]
        old_stamp = stamps_activity[0]
        for current_cell,current_timestamp in zip(cells_activity[1:], stamps_activity[1:]):
            if current_cell != old_cell:
                time_stayed_in_cells[old_cell] += current_timestamp-old_stamp
                old_cell = current_cell
                old_stamp = current_timestamp
                
                if time_stayed_in_cells[old_cell] < activity_stay_time:
                    time_stayed_in_cells[old_cell] = 0
        
            else: 
                time_stayed_in_cells[old_cell] += current_timestamp-old_stamp
                old_cell = current_cell
                old_stamp = current_timestamp
    
        for cell in time_stayed_in_cells:
            if time_stayed_in_cells[cell] >= activity_stay_time and cell not in nb_activity_cells and cell != home_cell:
                nb_activity_cells.append(cell)
        nb_activity_cells.sort(key=lambda c: cells_activity.count(c), reverse=True)
        cell_times   = [time_stayed_in_cells[c] for c in nb_activity_cells]
        cell_records = [records_per_cell[c]      for c in nb_activity_cells]

        candidates_activity = {cellid for cellid in time_stayed_in_cells.keys() if time_stayed_in_cells[cellid] >= activity_stay_time and cellid != home_cell}

        for cell in cells_activity:
            if cell in candidates_activity:
                Cell_Activity = cell # first cell found is the Cell_Activity
                break

        if Cell_Activity is None:
            reason_None.append("Not_stayed_enough_in_cells")


    # ====== #
    # Exit 3 #
    # ====== #
    return nb_activity_cells, reason_None, useful_merge_count_300, len(nb_activity_cells), cell_times, cell_records

def get_user_cells_no_cont_no_gap(cells_activity, stamps_activity, useful_merge_count_300, activity_stay_time : int | None = 18000, merge_func : Callable[[str], str] = lambda x: x, cells_full=None, stamps_full=None):
    """
    Returns the home and activity cell of the user or None if these cells don't exist for the user ans the number of activity cells (cells in which the user stayed more than 300 minutes in continued time) and the reason why the cell is None if it is the case.

    A first pass is done to look for these cells depending on the cellid, if not successful a second pass is made while merging
    cells into base station. A global counter is used throughout all the users to see how impactful this second pass is.
    """

    # ======= #
    # Phase 0 # Init variables
    # ======= #
    Cell_Activity = None
    reason_None = []

    nb_activity_cells = []
    home_cell = get_home_cell(cells_full, stamps_full, merge_func=merge_func) if cells_full is not None else None
    
    n_activity = len(cells_activity)
    if n_activity < 2:
        Cell_Activity = ""
        reason_None.append("Not_enough_records")
    
    # ====== #
    # Exit 1 # if not enough records
    # ====== # 
    if Cell_Activity == "":
        return nb_activity_cells, reason_None, useful_merge_count_300, len(nb_activity_cells), [], []
    
    # ======= #
    # Phase 1 # First search of cells
    # ======= #
    
    # First search for activity cell

    if Cell_Activity is None:
        # Check for cells having 300+ minutes of stay time (18000s)
        time_stayed_in_cells = Counter()
        records_per_cell     = Counter(cells_activity)
        old_cell = cells_activity[0]
        old_stamp = stamps_activity[0]
        for current_cell,current_timestamp in zip(cells_activity[1:], stamps_activity[1:]):
            if current_timestamp-old_stamp < 4*3600 + 600: # if the time between two records is more than 4 hours, we consider that the user has disconnected and reconnected and we reset the time stayed in cells
                time_stayed_in_cells[old_cell] += current_timestamp-old_stamp
                old_cell = current_cell
                old_stamp = current_timestamp
                
            else:
                old_cell = current_cell
                old_stamp = current_timestamp
    
        for cell in time_stayed_in_cells:
            if time_stayed_in_cells[cell] >= activity_stay_time and cell not in nb_activity_cells and cell != home_cell:
                nb_activity_cells.append(cell)
        nb_activity_cells.sort(key=lambda c: cells_activity.count(c), reverse=True)
        cell_times   = [time_stayed_in_cells[c] for c in nb_activity_cells]
        cell_records = [records_per_cell[c]      for c in nb_activity_cells]

        candidates_activity = {cellid for cellid in time_stayed_in_cells.keys() if time_stayed_in_cells[cellid] >= activity_stay_time and cellid != home_cell}

        for cell in cells_activity:
            if cell in candidates_activity:
                Cell_Activity = cell # first cell found is the Cell_Activity
                break

        if Cell_Activity is None:
            reason_None.append("Not_stayed_enough_in_cells")


    # ====== #
    # Exit 3 #
    # ====== #
    return nb_activity_cells, reason_None, useful_merge_count_300, len(nb_activity_cells), cell_times, cell_records

def get_user_cells_no_cont_gap(cells_activity, stamps_activity, useful_merge_count_300, activity_stay_time : int | None = 18000, merge_func : Callable[[str], str] = lambda x: x, cells_full=None, stamps_full=None):
    """
    Returns the home and activity cell of the user or None if these cells don't exist for the user ans the number of activity cells (cells in which the user stayed more than 300 minutes in continued time) and the reason why the cell is None if it is the case.

    A first pass is done to look for these cells depending on the cellid, if not successful a second pass is made while merging
    cells into base station. A global counter is used throughout all the users to see how impactful this second pass is.
    """

    # ======= #
    # Phase 0 # Init variables
    # ======= #
    Cell_Activity = None
    reason_None = []

    nb_activity_cells = []
    home_cell = get_home_cell(cells_full, stamps_full, merge_func=merge_func) if cells_full is not None else None
    
    n_activity = len(cells_activity)
    if n_activity < 2:
        Cell_Activity = ""
        reason_None.append("Not_enough_records")
    
    # ====== #
    # Exit 1 # if not enough records
    # ====== # 
    if Cell_Activity == "":
        return nb_activity_cells, reason_None, useful_merge_count_300, len(nb_activity_cells), [], []
    
    # ======= #
    # Phase 1 # First search of cells
    # ======= #
    
    # First search for activity cell

    if Cell_Activity is None:
        # Check for cells having 300+ minutes of stay time (18000s)
        time_stayed_in_cells = Counter()
        records_per_cell     = Counter(cells_activity)
        old_cell = cells_activity[0]
        old_stamp = stamps_activity[0]
        for current_cell,current_timestamp in zip(cells_activity[1:], stamps_activity[1:]):
            time_stayed_in_cells[old_cell] += current_timestamp-old_stamp
            old_cell = current_cell
            old_stamp = current_timestamp

        for cell in time_stayed_in_cells:
            if time_stayed_in_cells[cell] >= activity_stay_time and cell not in nb_activity_cells and cell != home_cell:
                nb_activity_cells.append(cell)
        nb_activity_cells.sort(key=lambda c: cells_activity.count(c), reverse=True)
        cell_times   = [time_stayed_in_cells[c] for c in nb_activity_cells]
        cell_records = [records_per_cell[c]      for c in nb_activity_cells]

        candidates_activity = {cellid for cellid in time_stayed_in_cells.keys() if time_stayed_in_cells[cellid] >= activity_stay_time and cellid != home_cell}

        for cell in cells_activity:
            if cell in candidates_activity:
                Cell_Activity = cell # first cell found is the Cell_Activity
                break

        if Cell_Activity is None:
            reason_None.append("Not_stayed_enough_in_cells")


    # ====== #
    # Exit 3 #
    # ====== #
    return nb_activity_cells, reason_None, useful_merge_count_300, len(nb_activity_cells), cell_times, cell_records

METHODS = {
    "cont_no_gap"    : get_user_cells_cont_no_gap,
    "cont_gap"       : get_user_cells_cont_gap,
    "no_cont_no_gap" : get_user_cells_no_cont_no_gap,
    "no_cont_gap"    : get_user_cells_no_cont_gap,
}


_NO_GAP_FUNCS   = {get_user_cells_cont_no_gap, get_user_cells_no_cont_no_gap}
_GAP_THRESHOLD  = 4 * 3600 + 600  # 4 h 10 min in seconds


def process_user_activity(cells : list[str], stamps : list[int],
                          activity_start : int = 18000, activity_end : int = 68400,
                          useful_merge_count_300 : int = 0,
                          merge_func : Callable[[str], str] = lambda x: x,
                          get_cells_func : Callable = get_user_cells_cont_no_gap):
    """
    Determine the activity cells for a given user/day.
    The merge is applied to the cells before calling get_cells_func.
    Boundary continuity: if the record just before (resp. after) the activity window
    has the same merged cell as the first (resp. last) in-window record, a synthetic
    record is injected at activity_start (resp. activity_end) so that the time already
    spent at that cell within the window is correctly counted.
    For no-gap methods the injection is skipped when the inter-record gap exceeds
    _GAP_THRESHOLD; gap methods always inject.
    """
    n_records = len(cells)

    if n_records == 1:
        return None, ["1_record"], useful_merge_count_300, 0, [], []

    cells_activity, stamps_activity, pre_cell, pre_stamp, post_cell, post_stamp = separate_day(
        cells=cells, stamps=stamps,
        activity_start=activity_start, activity_end=activity_end
    )

    cells_activity_merged = list(get_base_stations_list(cells_activity, merge_func=merge_func))
    stamps_activity       = list(stamps_activity)

    use_gap_check = get_cells_func in _NO_GAP_FUNCS

    # ── Start boundary ────────────────────────────────────────────────────────
    # If the last pre-window record is on the same cell as the first in-window
    # record, inject a synthetic anchor at activity_start so that the time
    # [activity_start → first_stamp] is counted for that cell.
    if pre_cell is not None and cells_activity_merged:
        pre_merged = merge_func(pre_cell)
        if pre_merged == cells_activity_merged[0]:
            gap = stamps_activity[0] - pre_stamp
            if not use_gap_check or gap < _GAP_THRESHOLD:
                cells_activity_merged.insert(0, pre_merged)
                stamps_activity.insert(0, activity_start)

    # ── End boundary ──────────────────────────────────────────────────────────
    # Symmetric: if the first post-window record continues on the same cell as
    # the last in-window record, inject a synthetic anchor at activity_end so
    # that the time [last_stamp → activity_end] is counted.
    if post_cell is not None and cells_activity_merged:
        post_merged = merge_func(post_cell)
        if post_merged == cells_activity_merged[-1]:
            gap = post_stamp - stamps_activity[-1]
            if not use_gap_check or gap < _GAP_THRESHOLD:
                cells_activity_merged.append(post_merged)
                stamps_activity.append(activity_end)

    activity_cells, reason_None, useful_merge_count_300, nb_activity_cells, cell_times, cell_records = get_cells_func(
        cells_activity_merged, stamps_activity,
        useful_merge_count_300, activity_stay_time=18000,
        merge_func=merge_func, cells_full=cells, stamps_full=stamps
    )

    return activity_cells, reason_None, useful_merge_count_300, nb_activity_cells, cell_times, cell_records


# ============ #
# MAIN SCRIPT  #  4 methods × 3 merges = 12 CSV files
# ============ #

def _empty_result_dict():
    return {"user_id": [], "day": [], "activity_cells": [],
            "reason_None": [], "period": [], "working_period": [],
            "nb_activity_cells": [], "cell_times": [], "cell_records": []}


for merge_name, merge_func in MERGE.items():
    for method_name, get_cells_func in METHODS.items():

        OUTPUT = INTERMEDIATE_RESULT / f"classified_dataset_{method_name}_merge_{merge_name}.csv"

        useful_merge_count_300 = 0
        result_days = _empty_result_dict()

        for file in tqdm.tqdm(files, desc=f"{merge_name}/{method_name}"):
            day = get_day(file)
            with open(file, mode='r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f, delimiter=';')
                for line in reader:
                    user_id    = line[0]
                    user_cells = [c  for c  in line[8::2] if c]
                    user_stamps= [int(ts) for ts in line[9::2] if ts]

                    for label, (activity_start, activity_end) in ACTIVITY_PERIOD.items():
                        activity_cells, reason_None, useful_merge_count_300, nb_activity_cells, cell_times, cell_records = \
                            process_user_activity(
                                cells=user_cells, stamps=user_stamps,
                                activity_start=activity_start, activity_end=activity_end,
                                useful_merge_count_300=useful_merge_count_300,
                                merge_func=merge_func,
                                get_cells_func=get_cells_func
                            )

                        result_days["user_id"].append(user_id)
                        result_days["day"].append(day)
                        result_days["activity_cells"].append(activity_cells)
                        result_days["reason_None"].append(reason_None)
                        result_days["period"].append(label)
                        result_days["working_period"].append((activity_start, activity_end))
                        result_days["nb_activity_cells"].append(nb_activity_cells)
                        result_days["cell_times"].append(cell_times)
                        result_days["cell_records"].append(cell_records)

        print(f"[{merge_name}/{method_name}] merge useful for {useful_merge_count_300} cases")

        pd.DataFrame(result_days).to_csv(OUTPUT, sep=";", header=True, index=False)