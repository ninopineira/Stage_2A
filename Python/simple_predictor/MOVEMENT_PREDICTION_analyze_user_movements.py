"""This script aims to show how user behaves when they move and when they stay in the same spot, in order to see how to
create nice features that would have a meaning to predict user movements.

We only consider users with at least 6 records as usual because less records is not relevant to predict anything.
One of the question is to consider cells or base station.


--> Analyze the distribution of the first and last movement of the user of the day
    --> for the last movement, the stat is computed only on users where they end in the same cell as they started
    --> The merged version is used as we want to capture true movements


Issues that we most surely remains unsolved :
- User of type  A - A - A - A - B - C - C - C - B - A - A : the movement is so short that we don't have a clue when it will happens
and it stops even before we have enough clue to say the user is moving
"""
import tqdm
import csv 
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import utils

def get_user_first_and_last_move(files_merged_2g3g):
    
    all_first_move_time_global = []
    all_last_move_time_global  = []
    n_user_go_back_home = 0
    
    all_first_move_time_all_day = []
    all_last_move_time_all_day = []
    
    day_list = []
    weekend_list = []
    
    for file in files_merged_2g3g:
        
        day = utils.get_day(file)
        weekend = utils.is_weekend(day)
        day_list.append(day)
        weekend_list.append(weekend)
        
        
        all_first_move_time_single_day = []
        all_last_move_time_single_day = []
        
        with open(file, mode='r') as f:
            reader = csv.reader(f, delimiter=";")
            for user in tqdm.tqdm(reader):
                n_records = int(user[7])
                if MAX_RECORDS >= n_records >= MIN_RECORDS: # At least 6 records
                    cells = user[8::2]
                    set_cell = set(cells)
                    
                    if len(set_cell) > 1: # At least one move (2 different cells)
                        index = 0
                        first_cell = cells[index]
                        timestamps = [int(ts) for ts in user[9::2]]
                        
                        while index < n_records and cells[index] == first_cell:
                            index += 1   
                        
                        time_first_move = timestamps[index]
                        all_first_move_time_global.append(time_first_move)
                        all_first_move_time_single_day.append(time_first_move)
                        
                        
                        last_cell = cells[-1]
                        if last_cell == first_cell:
                            n_user_go_back_home += 1
                            index = 2
                            
                            while index < n_records and cells[-index] == first_cell:
                                index += 1
                            
                            time_last_move = timestamps[-index]
                            all_last_move_time_global.append(time_last_move)
                            all_last_move_time_single_day.append(time_last_move)

        all_first_move_time_all_day.append(np.array(all_first_move_time_single_day) / 3600)
        all_last_move_time_all_day.append(np.array(all_last_move_time_single_day) / 3600)
            
    
    print("First and last mean hour")
    print(np.mean(all_first_move_time_global)/3600)
    print(np.mean(all_last_move_time_global)/3600)
    
    print("First move description")
    all_first_move_time_global = pd.Series(all_first_move_time_global)
    print(all_first_move_time_global.describe())
    
    print("Last move description")
    all_last_move_time_global = pd.Series(all_last_move_time_global)
    print(all_last_move_time_global.describe())
    
    
    all_first_move_time_global = np.array(all_first_move_time_global) / 3600
    all_last_move_time_global = np.array(all_last_move_time_global) / 3600
    plot_histo(data = all_first_move_time_global, output_name="all_first_move_time_distribution", color = "green", name = "Distribution of first timestamp of user \nwith [cellID] != [cellID of first record]\nAll days")
    plot_histo(data = all_last_move_time_global , output_name="all_last_move_time_distribution" , color = "orange", name = "Distribution of last timestamp of user \nwith [cellID] != [cellID of first record]\nAll days")
    plot_histo(data = all_first_move_time_all_day , output_name="all_day_first_move_time_distribution", day_list = day_list, weekend_list = weekend_list)
    plot_histo(data = all_last_move_time_all_day  , output_name="all_day_last_move_time_distribution", day_list = day_list, weekend_list = weekend_list)
                        
                        
def plot_histo(data, output_name : str, color : str | None = None, day_list = None, weekend_list = None, name : str | None = None):
    
    if isinstance(data[0], np.float64):
        
        plt.hist(data, bins = 48, range = (0,24), color=color)
        plt.title(name)
        plt.xlabel("Timestamp (h)")
        plt.ylabel("Number of user")
        plt.savefig(MAIN_DIR / f"results/predictions/movement_prediction/analyze_plots/{output_name}.png", dpi=300)
        plt.close()
        
    else:
        n_plot = len(data) # 15
        nrows, ncols = n_plot // 5, n_plot // 2 
        fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharey = True, figsize=(18, 12))
        
        axes_flatten = axes.flatten()
        plot_colors = ["lightcoral","darkorange","gold","chartreuse","seagreen","aquamarine","slategrey"]
        
        
        for i in range(n_plot):
            ax = axes_flatten[i]
            ax : plt.Axes
            color = plot_colors[i % 7]
                        
            title_color = 'red' if weekend_list[i] else "black"
            ax.hist(data[i], bins = 48, range = (0,24), color = color)
            ax.set_title(f"{day_list[i]}", color=title_color)
            ax.set_xlabel("Timestamp (h)")
            ax.set_ylabel("Number of user")
        
        for i in range(n_plot,n_plot+ncols):
            ax = axes_flatten[i]
            ax : plt.Axes
            ax.axis('off')
        
        fig.suptitle("Distribution of first timestamp of user\nwhere [cellID] != [cellID of first record]")
        
        plt.tight_layout()
        fig.savefig(MAIN_DIR / f"results/predictions/movement_prediction/analyze_plots/{output_name}.png", dpi=300)
        plt.close()    

                        
                        
                        

if __name__ == "__main__":
    MIN_RECORDS = 6
    MAX_RECORDS = 512
    BASE_POINT = (50.0,12.0)
    
    MAIN_DIR = Path(__file__).parent.parent.parent
    DATABASE_DIR = MAIN_DIR / "Database/no_duplicate_merge_2g3g"
    files = DATABASE_DIR.glob("*.csv")

    CELLS = MAIN_DIR / "Database/cells/cells_of_dataset_cd_142.csv"
    # df = pd.read_csv(CELLS, delimiter=";")
    # coords_cache = dict(zip(df['cellid'], zip(df['lat'], df['lon'])))
    
    get_user_first_and_last_move(files_merged_2g3g=files)