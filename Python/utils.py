import json
import csv
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from geopy.distance import geodesic

# import sys
# sys.path.append("C:/Program Files (x86)/QGIS Brighton/apps/qgis/python")
# from qgis.core import (
#     QgsApplication,
#     QgsProject,
#     QgsVectorLayer,
#     QgsFeature,
#     QgsGeometry,
#     QgsPointXY
# )


# =============== #
# Sets of columns #
# =============== #
COLUMN_NAMES = ["user_id","age","gender","unkown_numbers","letters","BS1","BS2","n_records"]

# ===== #
# Paths #
# ===== #
ORIGINAL_CSV_DIR = Path("../Database/csv")


# =============== #
# File management #
# =============== #
def open_csv_as_dataframe(path, give_names_to_column=False) -> pd.DataFrame:
    """Ouvre un CSV et renomme proprement les premières colonnes."""
    
    # Chargement du DataFrame
    if has_header(path):
        df = pd.read_csv(path, sep=None, index_col=False, engine="python")
    else:
        df = pd.read_csv(path, sep=None, header=None, index_col=False, engine="python")

    if give_names_to_column:
        mapping = dict(zip(df.columns, COLUMN_NAMES))
        df = df.rename(columns=mapping)
        
    return df

def load_station_list(path) -> pd.DataFrame :
    df = pd.read_csv(path, sep=";", index_col=False)
    return df

def save_to_json(output_path, data : dict):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

# ============= #
# Data analysis #
# ============= #
def separate_data_by_letter_code(dataframe : pd.DataFrame, column_name : str | None = "letters") -> dict : 
    """Separates the dataset depending on the column with the 4 letters codes for further analysis
    
    returns : letters_seperation_dict with keys = ['UNKNOWN', 'CETR', 'NUDM', nan, 'CENU', 'CEBU', 'CEZB', 'BUNE', 'NUDU', 'NUNU']
                                    and values are the fitered dataframe depending on the letter code
    """

    letters = dataframe[column_name].tolist()
    letters = list(set(letters)) # [nan, 'CETR', 'NUDM', nan, 'CENU', 'CEBU', 'CEZB', 'BUNE', 'NUDU', 'NUNU']
    
    # I need to remove the nan values because it causes problem and I replace it with a true str : "UNKNOWN"
    i = 0
    while i<len(letters) and isinstance(letters[i],str):
        i+=1
    letters[i] = "UNKNOWN"
    
    letters_seperation_dict = dict.fromkeys(letters)

    # Extracting the data depending on their "letters" column to see if there is a pattern with the rest of the line
    for key in list(letters_seperation_dict.keys()):
        
        if key == "UNKNOWN":
            letters_seperation_dict[key] = {"df" : dataframe[dataframe["letters"].isna()]}
            
        else:
            letters_seperation_dict[key] = {"df" : dataframe[dataframe["letters"] == key]}
    
    return letters_seperation_dict

def count_null_values(data) :
    """Count the number of null values for a full dataframe or just a column depending of the input"""
    return len(data) - data.count()

def has_header(file, nrows=20):
    
    """Check if dataframe has header
    returns True if headers are present else False
    """
    
    df = pd.read_csv(file, header=None, nrows=nrows)
    df_header = pd.read_csv(file, nrows=nrows)
    
    return tuple(df.dtypes) != tuple(df_header.dtypes)

def find_all_stations(csv_directory : Path) -> list:
    """
    returns every station found for the given csv file in the directory placed as inputs (non-recursive)
    """
    
    files = [file for file in csv_directory.glob("*.csv")]
    
    unique_stations = set()
    
    for file in files:
        with open(file, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                for value in row:
                    if value != '' and '-' not in value and not value.isnumeric():
                        unique_stations.add(value)
    
    return list(unique_stations)


# ========== #
# QGIS UTILS #
# ========== #
def draw_trajectory(stations : list):
    """
    Draw the trajectory for visualization on the qgis map using GeoJSON notation
    
    base_stations : list of (str, (float,float)) 
    """
    
    features = []

    # ligne
    features.append({
        "type": "Feature",
        "properties": {"type": "trajet"},
        "geometry": {
            "type": "LineString",
            "coordinates": [pos for name, pos in stations]
        }
    })

    # points
    for name, (lon, lat) in stations:
        features.append({
            "type": "Feature",
            "properties": {"name": name, "type": "point"},
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open("trajet.geojson", "w") as f:
        json.dump(geojson, f, indent=2)


# ====== #
# Others #
# ====== #
def get_day(filepath : Path) -> str:
    return filepath.name.split("_")[0]

def is_weekend(day : str) -> bool:
    """
    Determine if day is part of a weekend or not.
    
    ONLY FORMAT ACCEPTED FOR INPUT IS : "YYYY-MM-DD"
    """
    split_day = day.split("-")
    day_object = datetime.date(year = int(split_day[0]),
                  month = int(split_day[1]),
                  day = int(split_day[2]))
    return day_object.weekday() > 4


def compute_distance_travelled_by_user(cellid_list : list[str], coords_cache : dict[str, tuple[float,float]]) -> float :
    """
    Compute the distance travelled by the user during one day based on the cellid records of the user.
    The computed distance is the sum of the distance between consecutive records whenever the cells are different
    
    If there is only one record : the distance is set to 0
    
    """
    
    distance_travelled = 0
    n_record = len(cellid_list)
    
    if n_record != 1:
        for i in range(len(cellid_list)-1):
            cell_start = cellid_list[i]
            cell_end = cellid_list[i+1]
            
            if cell_start != cell_end:
                lat_start,lon_start = coords_cache[cell_start]
                lat_end, lon_end    = coords_cache[cell_end]
                distance_travelled += convert_lat_lon_distance_to_meter(lat1 = lat_start, lat2 = lat_end, lon1 = lon_start, lon2 = lon_end)
            
    return distance_travelled

def convert_lat_lon_distance_to_meter(lat1 : float, lat2 : float, lon1 : float, lon2 : float) -> float :
    """
    Returns the distance between 2 points at the surface of the Earth in meters
    """    
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)

    distance = geodesic(point1, point2).meters
    return distance

def get_lat_lon_cell(dataframe : pd.DataFrame, cellid : str) :
    return dataframe[dataframe["cellid"] == cellid]["lat"].values[0], dataframe[dataframe["cellid"] == cellid]["lon"].values[0]

