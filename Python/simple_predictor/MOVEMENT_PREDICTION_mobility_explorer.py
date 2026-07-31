import csv
import os
import glob
from pathlib import Path
from geopy.distance import geodesic
import plotly.graph_objects as go
from datetime import datetime

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
MAIN_DIR = Path(__file__).parent.parent.parent

DATA_FOLDER   = MAIN_DIR / "Database/no_duplicate"          # Folder containing user CSV files
CELLS_CSV     = MAIN_DIR / "Database/cells/cd_142_cells.csv"     # CSV with cellid;lat;lon;x;y
MIN_RECORDS   = 6                 # Minimum number of records per user
CSV_DELIMITER = ";"               # Delimiter for user CSV files

# Column indices in user CSV rows
COL_USER_ID     = 0
COL_N_RECORDS   = 7
COL_CELLS_START = 8   # cells at indices 8, 10, 12, ...
COL_TS_START    = 9   # timestamps at indices 9, 11, 13, ...

# Timestamp format — adjust if needed (used to extract date for "first cell of the day")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"  # e.g. "2023-06-15 08:32:00"
# ─────────────────────────────────────────


def load_cells(cells_csv: str) -> dict:
    """Load cell coordinates. Returns {cellid: (lat, lon)}."""
    cells = {}
    with open(cells_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            cells[row['cellid'].strip()] = (float(row['lat']), float(row['lon']))
    return cells


def load_users(data_folder: str) -> list:
    """Load all CSV rows from the data folder."""
    pattern = os.path.join(data_folder, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {data_folder}")
    print(f"  Found {len(files)} file(s) in {data_folder}")

    rows = []
    for filepath in files:
        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=CSV_DELIMITER)
            for row in reader:
                if row:
                    rows.append(row)
    print(f"  {len(rows)} total rows loaded")
    return rows


def parse_date(ts: str) -> str:
    """Extract the date part of a timestamp string (YYYY-MM-DD)."""
    try:
        return datetime.strptime(ts.strip(), TIMESTAMP_FORMAT).strftime("%Y-%m-%d")
    except ValueError:
        # Fallback: take the first 10 chars if format differs
        return ts.strip()[:10]


def compute_distances(cells: list, timestamps: list, cell_coords: dict) -> list:
    """
    For each record, compute geodesic distance to the first cell seen.
    Returns a list of distances in km (None if cell not found in coords dict).
    """
    ref_cell = cells[0]
    coord_ref  = cell_coords[ref_cell]

    distances = []
    for cell in cells:
        if cell == ref_cell:
            distances.append(0.0)
        else:
            coord_cell = cell_coords[cell]
            distances.append(geodesic(coord_ref, coord_cell).km)
    return distances


def plot_user(user_id: str, all_rows: list, cell_coords: dict):
    """Find the user, compute distances, and display an interactive Plotly chart."""
    user_rows = [r for r in all_rows if r[COL_USER_ID].strip() == user_id.strip()]

    if not user_rows:
        print(f"  ✗ No user found with ID '{user_id}'")
        return

    row = user_rows[0]

    try:
        n_records = int(row[COL_N_RECORDS])
    except (ValueError, IndexError):
        print(f"  ✗ Could not read record count for user '{user_id}'")
        return

    if n_records < MIN_RECORDS:
        print(f"  ✗ User '{user_id}' only has {n_records} records (min: {MIN_RECORDS})")
        return

    cells      = row[COL_CELLS_START::2]
    timestamps = [int(ts) for ts in row[COL_TS_START::2]]

    # Trim to same length in case of misalignment
    n = min(len(cells), len(timestamps))
    cells, timestamps = cells[:n], timestamps[:n]

    if len(set(cells)) <= 1:
        print(f"  ✗ User '{user_id}' never moved (only 1 unique cell)")
        return

    distances = compute_distances(cells, timestamps, cell_coords)
    
    # Filter out None values for plotting
    plot_ts   = [ts for ts, d in zip(timestamps, distances) if d is not None]
    plot_dist = [d  for d in distances if d is not None]
    plot_cell = [c  for c, d in zip(cells, distances) if d is not None]

    print(f"  ✓ User '{user_id}' — {len(plot_ts)} records, "
          f"{len(set(cells))} unique cells, "
          f"max dist {max(plot_dist):.2f} km")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_ts,
        y=plot_dist,
        mode='lines+markers',
        line=dict(color='#7c6fe0', width=2),
        marker=dict(color='#4ecdc4', size=7, symbol='circle'),
        hovertemplate=(
            '<b>Timestamp:</b> %{x}<br>'
            '<b>Distance:</b> %{y:.2f} km<br>'
            '<b>Cell:</b> %{text}<extra></extra>'
        ),
        text=plot_cell,
        name=user_id,
    ))

    fig.update_layout(
        title=dict(
            text=f"User <b>{user_id}</b> distance to first cell ({plot_cell[0]}) of the day over time",
            font=dict(size=15, color='#e8e6f0'),
            x=0.02,
        ),
        paper_bgcolor='#0e0e12',
        plot_bgcolor='#16161c',
        font=dict(family='monospace', color='#e8e6f0', size=11),
        xaxis=dict(
            title='Timestamp',
            color='#7a7890',
            gridcolor='rgba(255,255,255,0.05)',
            tickangle=-35,
            range=[0, 86400]
        ),
        yaxis=dict(
            title='Distance to first cell of the day (km)',
            color='#7a7890',
            gridcolor='rgba(255,255,255,0.05)',
            rangemode='tozero',
        ),
        hoverlabel=dict(
            bgcolor='#1e1e26',
            bordercolor='#7c6fe0',
            font=dict(family='monospace', size=11),
        ),
        margin=dict(l=80, r=30, t=60, b=80),
    )

    fig.show()


def main():
    print("\n── MOBILITY EXPLORER ─────────────────────────")
    print(f"  Data folder : {DATA_FOLDER}")
    print(f"  Cells CSV   : {CELLS_CSV}")
    print(f"  Min records : {MIN_RECORDS}")
    print("───────────────────────────────────────────────\n")

    print("Loading cell coordinates...")
    cell_coords = load_cells(CELLS_CSV)
    print(f"  {len(cell_coords)} cells loaded\n")

    print("Loading user data...")
    all_rows = load_users(DATA_FOLDER)
    print()

    while True:
        user_id = input("Enter user ID (or 'q' to quit): ").strip()
        if user_id.lower() == 'q':
            break
        if not user_id:
            continue
        plot_user(user_id, all_rows, cell_coords)
        print()


if __name__ == "__main__":
    main()
