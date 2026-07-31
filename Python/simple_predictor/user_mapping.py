from utils import HelperData;from pathlib import Path
MAIN_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = MAIN_DIR / f"Database/no_duplicate"
OUTPUT = MAIN_DIR / f"results/simple_predictor/user_mapping.json"
helper = HelperData()
helper.map_users_to_day_and_file_row(dataset_filepath = DATASET_DIR, output_path = OUTPUT)