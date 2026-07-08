import csv
import re
from pathlib import Path

MAIN_DIR  = Path(__file__).parent.parent
CELLS_CSV = MAIN_DIR / "Database/cells/cd_142_cells.csv"

# First character of the cell ID encodes the radio technology:
#   B, D  →  2G (GSM / EDGE)
#   U, V  →  3G (UMTS / HSPA)
TECH_2G = {"B", "D"}
TECH_3G = {"U", "V"}


def base_station(cell_id: str) -> str:
    """Return the base-station name: letter prefix with the first char removed."""
    m = re.match(r"([a-zA-Z]+)", cell_id)
    return m.group(1)[1:] if m else ""


def load_cells(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def find_dual_technology_stations(cells: list[dict]) -> dict[str, dict]:
    """
    Return a dict keyed by base-station name, containing only stations
    that have at least one 2G cell AND at least one 3G cell.
    """
    stations: dict[str, dict] = {}

    for row in cells:
        cid  = row["cellid"]
        tech = cid[0]
        base = base_station(cid)
        if not base or tech not in (TECH_2G | TECH_3G):
            continue

        if base not in stations:
            stations[base] = {"2g": [], "3g": []}

        if tech in TECH_2G:
            stations[base]["2g"].append(cid)
        elif tech in TECH_3G:
            stations[base]["3g"].append(cid)

    return {b: v for b, v in stations.items() if v["2g"] and v["3g"]}


if __name__ == "__main__":
    cells = load_cells(CELLS_CSV)
    print(f"Total cells in database: {len(cells)}")

    dual = find_dual_technology_stations(cells)

    print(f"\nBase stations with both 2G and 3G cells: {len(dual)}\n")
    print(f"{'Station':<12}  {'2G cells':<40}  {'3G cells'}")
    print("-" * 90)
    for base, techs in sorted(dual.items()):
        g2 = ", ".join(sorted(techs["2g"]))
        g3 = ", ".join(sorted(techs["3g"]))
        print(f"{base:<12}  {g2:<40}  {g3}")

    # Flat list of all cell IDs belonging to dual-technology stations
    all_dual_cells = sorted(
        cid
        for techs in dual.values()
        for cid in techs["2g"] + techs["3g"]
    )
    print(f"\nTotal cell IDs in dual-technology stations: {len(all_dual_cells)}")
