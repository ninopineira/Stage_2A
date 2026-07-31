# Creates a cellid map to convert the cellids str into ints
# 
# Also added the creation of a mapping between the days and basic ints (0 to 14)
# I didn't encode the days in a more sophisticated way because of all the analysis made during 2 months ;
# no pattern was noticed between 2 thursdays and as we have only 2 weeks, even if the model would be capable to
# detect patterns, it's a too short time period to infer anything. Additionnaly, as we can see for Friday 21-03-2014,
# some problems in the data provided by the companies (see the Timestamp_heatmap.png), it could mislead the model
# learning wrong behaviour and degrading the global performances.  

import pandas as pd
import json
from pathlib import Path

MAIN_DIR = Path(__file__).parent.parent.parent
cells_path = MAIN_DIR / "Database/cells/cd_142_cells.csv"
OUTPUT_DIR = MAIN_DIR / "results/predictions/deep_learning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(cells_path, sep=";", index_col=False)
assert len(df.cellid.unique()) == len(df), "All rows should be unique"
cells = df.cellid.tolist()
cells = sorted(cells)

# For simple training for time weight
map_dict = {cell : i for i,cell in enumerate(cells)}
with open(OUTPUT_DIR / "cell_map.json", mode="w", encoding='utf-8') as f:
    json.dump(map_dict,f)

# For Transformer
map_dict = {cell : i for i,cell in enumerate(cells, start=1)}
with open(OUTPUT_DIR / "cell_map_start_1.json", mode="w", encoding='utf-8') as f:
    json.dump(map_dict,f)


# DAYS
OUTPUT_DAYS = OUTPUT_DIR / "day_map.json"
DAYS = ["2014-03-12","2014-03-13","2014-03-14","2014-03-15","2014-03-16",
        "2014-03-17","2014-03-18","2014-03-19","2014-03-20","2014-03-21","2014-03-22","2014-03-23",
        "2014-03-24","2014-03-25","2014-03-26"]

r = {d : i for i,d in enumerate(DAYS)}
with open(OUTPUT_DAYS, mode="w") as f:
    json.dump(r,f)