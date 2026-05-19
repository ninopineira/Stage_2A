from pathlib import Path
import pandas as pd
import numpy as np

MAIN_DIR = Path(__file__).parent.parent.parent
INTERMEDIATE_PATH = MAIN_DIR / f"results/intermediate_result/classified_dataset.csv"

OUTPUT_DIR = MAIN_DIR / f"results/intermediate_result"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MERGE = {
    "no_merge" : MAIN_DIR / f"results/intermediate_result/classified_dataset_merge_no_merge.csv",
    "simple" : MAIN_DIR / f"results/intermediate_result/classified_dataset_merge_simple.csv",
    "2g3g" : MAIN_DIR / f"results/intermediate_result/classified_dataset_merge_2g3g.csv"
}

for merge_key, path in MERGE.items():

    df = pd.read_csv(path, sep=";", index_col=False, 
                    dtype={"user_id" : str,
                        "day" : str,
                        "Cell_Activity" : str,
                        "reason_None" : str,
                        "period" : str,
                        "working_period" : str,
                        "nb_activity_cells" : str})
    df = df.replace('',np.nan)

    data = df[(df["day"] == "2014-03-12") & (df["period"] == "Home_04h10-19h50_Activity_05h00-19h00")]
    az = data[["user_id","reason_None"]]
    az = az[az.reason_None != "['1_record']"]
    
    test = az[az.reason_None == "['Not_stayed_enough_in_cells']"]
    
    count = az.groupby(by=["reason_None"]).size()
    
    count.sort_values(ascending=False)
    


    # ================================= #
    # Décompte cell activity #
    # ================================= #

    print("Merged...")

    if not (OUTPUT_DIR / f"cell_activity_merged_count_{merge_key}.csv").exists():
        df_Cell_Activity_merged = df[df["Cell_Activity"].notna()]
        group_Cell_Activity_merged = df_Cell_Activity_merged.groupby(by=['day','period', 'nb_activity_cells'])
        detail_group_Cell_Activity_merged = group_Cell_Activity_merged.size().unstack(level=['period', 'nb_activity_cells'], fill_value=0)
        detail_group_Cell_Activity_merged.to_csv(OUTPUT_DIR / f"cell_activity_merged_count_{merge_key}.csv", sep=";", index_label=True)
