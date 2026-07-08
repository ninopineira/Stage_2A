"""
Delete old entropy plot files that used deprecated naming conventions.

Old patterns (to delete):
  - *_classic*.png       (old English name, lowercase)
  - *_normalised*.png    (old English name, lowercase)
  - *_classique*.png     (French name)
  - *_matin_*.png        (French period name)
  - *_jour_*.png         (French period name)
  - *_soir_*.png         (French period name)

Current naming uses:
  - *_Uncorrelated_temporal_entropy_*.png
  - *_Relative_entropy_*.png

Set DRY_RUN = False to actually delete the files.
"""

from pathlib import Path

PLOTS_DIR = Path(__file__).parent.parent.parent / "results/plots"
DRY_RUN   = False  # ← set to True to preview without deleting

OLD_PATTERNS = [
    "*_classic*.png",
    "*_normalised*.png",
    "*_classique*.png",
    "*_matin_*.png",
    "*_jour_*.png",
    "*_soir_*.png",
]

# Collect all matching files (recursively, across day subfolders)
to_delete: list[Path] = []
for pattern in OLD_PATTERNS:
    to_delete.extend(PLOTS_DIR.rglob(pattern))

# Deduplicate (a file could match multiple patterns)
to_delete = sorted(set(to_delete))

if not to_delete:
    print("No old files found.")
else:
    print(f"{'[DRY RUN] ' if DRY_RUN else ''}Found {len(to_delete)} file(s) to delete:\n")
    for f in to_delete:
        print(f"  {f.relative_to(PLOTS_DIR)}")

    if not DRY_RUN:
        for f in to_delete:
            f.unlink()
        print(f"\nDeleted {len(to_delete)} file(s).")
    else:
        print("\nSet DRY_RUN = False to actually delete these files.")
