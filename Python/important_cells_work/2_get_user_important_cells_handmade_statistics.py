from pathlib import Path
import pandas as pd
import numpy as np

MAIN_DIR = Path(__file__).parent.parent.parent
INTERMEDIATE_PATH = MAIN_DIR / f"results/intermediate_result/classified_dataset.csv"

OUTPUT_DIR = MAIN_DIR / f"results/intermediate_result"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MERGE = {
    "simple" : MAIN_DIR / f"results/intermediate_result/classified_dataset_merge_simple.csv",
    "2g3g" : MAIN_DIR / f"results/intermediate_result/classified_dataset_merge_2g3g.csv"
}

for merge_key, path in MERGE.items():

    df = pd.read_csv(path, sep=";", index_col=False, 
                    dtype={"user_id" : str,
                        "day" : str,
                        "user_presence_classification" : int,
                        "Cell_Home" : str,
                        "Cell_Activity" : str,
                        "Same_Home_Activity" : bool,
                        "merged_Cell_Home" : str,
                        "merged_Cell_Activity" : str,
                        "merged_Same_Home_Activity" : bool,
                        "reason_None" : str,
                        "period" : str,
                        "working_period" : str,
                        "nb_activity_cells" : str})
    df = df.replace('',np.nan)

    data = df[(df["day"] == "2014-03-12") & (df["period"] == "Home_04h10-19h50_Activity_05h00-19h00")]
    az = data[data.user_presence_classification == 4][["user_id","user_presence_classification","reason_None"]]
    az = az[az.reason_None != "['1_record']"]
    
    test1 = az[az.reason_None == "['No_morning', 'No_evening']"]
    test2 = az[az.reason_None == "['No_morning', 'No_evening', 'Not_stayed_enough_in_cells', 'Not_stayed_enough_in_cells']"]
    
    count = az.groupby(by=["reason_None"]).size()
    
    count.sort_values(ascending=False)
    

    # ============== #
    # Classification #
    # ============== #
    print("Classification by presence...")
    if not (OUTPUT_DIR / f"classification_count.csv").exists():        
        group_presence = df.groupby(by=["day","user_presence_classification","period"])
        detail_group_presence = group_presence.size().unstack(fill_value=0)
        detail_group_presence.to_csv(OUTPUT_DIR / f"classification_count.csv", sep=";", index_label=True)

    # ================================= #
    # Décompte cell home, cell activity #
    # ================================= #
    print("Base...")
    if not (OUTPUT_DIR / f"cell_home_count.csv").exists():
        df_Cell_Home = df[df["Cell_Home"].notna()]
        group_Cell_Home = df_Cell_Home.groupby(by=['day','period'])
        detail_group_Cell_Home = group_Cell_Home.size().unstack(fill_value=0)
        detail_group_Cell_Home.to_csv(OUTPUT_DIR / f"cell_home_count.csv", sep=";", index_label=True)
    
    if not (OUTPUT_DIR / f"cell_activity_count.csv").exists():
        df_Cell_Activity = df[df["Cell_Activity"].notna()]
        group_Cell_Activity = df_Cell_Activity.groupby(by=['day','period', 'nb_activity_cells'])
        detail_group_Cell_Activity = group_Cell_Activity.size().unstack(level=['period', 'nb_activity_cells'], fill_value=0)
        detail_group_Cell_Activity.to_csv(OUTPUT_DIR / f"cell_activity_count.csv", sep=";", index_label=True)

    print("Merged...")
    if not (OUTPUT_DIR / f"cell_home_merged_count_{merge_key}.csv").exists():
    
        df_Cell_Home_merged = df[(df["Cell_Home"].notna() | df["merged_Cell_Home"].notna())]
        group_Cell_Home_merged = df_Cell_Home_merged.groupby(by=['day','period'])
        detail_group_Cell_Home_merged = group_Cell_Home_merged.size().unstack(fill_value=0)
        detail_group_Cell_Home_merged.to_csv(OUTPUT_DIR / f"cell_home_merged_count_{merge_key}.csv", sep=";", index_label=True)

    if not (OUTPUT_DIR / f"cell_activity_merged_count_{merge_key}.csv").exists():
        df_Cell_Activity_merged = df[(df["Cell_Activity"].notna() | df["merged_Cell_Activity"].notna())]
        group_Cell_Activity_merged = df_Cell_Activity_merged.groupby(by=['day','period', 'nb_activity_cells'])
        detail_group_Cell_Activity_merged = group_Cell_Activity_merged.size().unstack(level=['period', 'nb_activity_cells'], fill_value=0)
        detail_group_Cell_Activity_merged.to_csv(OUTPUT_DIR / f"cell_activity_merged_count_{merge_key}.csv", sep=";", index_label=True)

    # ==================================================== #
    # Décompte des gens qui ont cell home == cell activity #
    # ==================================================== #    
    df_home_and_activity = df[(df["Cell_Home"].notna()) & (df["Cell_Activity"].notna())]
    df_diff_home_activity = df[(df["Cell_Home"].notna()) & (df["Cell_Activity"].notna()) & (df["Cell_Home"] != df["Cell_Activity"])]
    df_same_home_activity = df[(df["Cell_Home"].notna()) & (df["Cell_Activity"].notna()) & (df["Cell_Home"] == df["Cell_Activity"])]
    
    g1,g2,g3 = df_home_and_activity.groupby(by = ["day","period"]), df_diff_home_activity.groupby(by = ["day","period"]) , df_same_home_activity.groupby(by = ["day","period"])
    d1,d2,d3 = g1.size().unstack(fill_value=0), g2.size().unstack(fill_value=0), g3.size().unstack(fill_value=0)
    print(f"Home and activity {len(df_home_and_activity)}")
    print(d1["Home_04h10-19h50_Activity_05h00-19h00"])
    print(f"Home != activity {len(df_diff_home_activity)}")
    print(d2["Home_04h10-19h50_Activity_05h00-19h00"])
    print(f"Home == activity {len(df_same_home_activity)}")
    print(d3["Home_04h10-19h50_Activity_05h00-19h00"])
    
    
    
    df_same_home_activity_merged = \
        df[( (df["merged_Cell_Home"].notna()) & (df["merged_Cell_Activity"].notna()) & (df["merged_Cell_Home"] == df["merged_Cell_Activity"]) ) | \
        ((df["Cell_Home"].notna()) & (df["Cell_Activity"].notna()) & (df["Cell_Home"] == df["Cell_Activity"]))]
    group_same_home_activity_merged = df_same_home_activity_merged.groupby(by = ["day","period"])
    detail_group_same_home_activity_merged = group_same_home_activity_merged.size().unstack(fill_value=0)
    detail_group_same_home_activity_merged["Home_04h10-19h50_Activity_05h00-19h00"]
    
    print("=======================================================================")
    print(f"Home activity MERGED : {merge_key} \n {detail_group_same_home_activity_merged["Home_04h10-19h50_Activity_05h00-19h00"]}")
    print("=======================================================================")
    
    
    # Autre décompte # 
    same = df[df["Same_Home_Activity"] | df["merged_Same_Home_Activity"]].groupby(by=["day","period"]).size().unstack(fill_value=0)
    df[(df["Cell_Activity"].isna() & df["merged_Cell_Activity"].notna())].groupby(by=["day","period"]).size().unstack(fill_value=0)["Home_04h10-19h50_Activity_05h00-19h00"]