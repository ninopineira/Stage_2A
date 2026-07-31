from pathlib import Path
import csv

MAIN_DIR  = Path(__file__).parent.parent
INPUT_DIR = MAIN_DIR / "Database/no_duplicate"

total        = 0
both_present = 0
same         = 0
only_home    = 0
only_act     = 0
neither      = 0

for file in sorted(INPUT_DIR.glob("*.csv")):
    with open(file, mode="r", encoding="utf-8", newline="") as f:
        for line in csv.reader(f, delimiter=";"):
            if len(line) < 7:
                continue
            home = line[5].strip()
            act  = line[6].strip()
            total += 1

            has_home = bool(home)
            has_act  = bool(act)

            if has_home and has_act:
                both_present += 1
                if home == act:
                    same += 1
            elif has_home:
                only_home += 1
            elif has_act:
                only_act += 1
            else:
                neither += 1

print(f"\n{'═'*60}")
print(f"  HOME CELL == ACTIVITY CELL  (dataset only)")
print(f"{'═'*60}")
print(f"  Total user×day records   : {total:>10,}")
print(f"  Both cells present       : {both_present:>10,}  ({both_present/total:.1%})")
print(f"  Same cell (home == act)  : {same:>10,}  ({same/total:.1%} of total, "
      f"{same/both_present:.1%} of both-present)")
print(f"  Different cells          : {both_present-same:>10,}  ({(both_present-same)/total:.1%})")
print(f"  Home only (no act cell)  : {only_home:>10,}  ({only_home/total:.1%})")
print(f"  Act only  (no home cell) : {only_act:>10,}  ({only_act/total:.1%})")
print(f"  Neither                  : {neither:>10,}  ({neither/total:.1%})")
print(f"{'═'*60}\n")
