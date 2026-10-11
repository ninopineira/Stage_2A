# Volume du dataset jour par jour, en un seul passage sur Database/no_duplicate.
# C'est la table de référence des rapports (tab:perday) : toutes les tailles de population
# citées en découlent.
#
# Le même passage mesure aussi ce qui justifie deux conventions du projet :
#   - les intervalles entre enregistrements consécutifs (pic à 4 h pile = mise à jour périodique
#     du téléphone au repos) et le nombre d'intervalles au-delà des trois marges de gap du dépôt ;
#   - l'effet des trois fusions de cellules (nombre de symboles, taux d'immobilité par utilisateur).
#
# Sorties :
#   results/intermediate_result/dataset_volume_by_day.csv   une ligne par jour + une ligne TOTAL
#   results/intermediate_result/dataset_volume_summary.csv  indicateurs globaux (gaps, fusions)
#   results/plots/records_per_user_day_hist.png             part des utilisateurs-jours ayant >= n records (log-log)
#   results/plots/records_per_minute_by_day.png             heat map minute x jour (anomalie du 21/03)

from collections import Counter
import csv
import datetime
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from utils import get_day, is_weekend

MAIN_DIR = Path(__file__).parent.parent
INPUT_DIR = MAIN_DIR / "Database/no_duplicate"
files = sorted(INPUT_DIR.glob("*.csv"))

INTERMEDIATE_PATH = MAIN_DIR / "results/intermediate_result"
INTERMEDIATE_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH = MAIN_DIR / "results/plots"
PLOT_PATH.mkdir(parents=True, exist_ok=True)

MAX_RECORDS = 512          # seuil au-delà duquel les lignes sont écartées (deep learning, move/stay)
MIN_RECORDS = 10           # population des taux d'immobilité par utilisateur (celle de Machin_learning/)
# Les trois marges de gap qui cohabitent dans le dépôt
GAP_LIMITS = {"4h+30s": 4 * 3600 + 30, "4h+60s": 4 * 3600 + 60, "4h10": 4 * 3600 + 600}
PERIODIC_UPDATE = (4 * 3600 - 60, 4 * 3600 + 60)   # [3 h 59 ; 4 h 01[
MINUTES_PER_DAY = 24 * 60
BIN_MINUTES = 10           # résolution de la heat map


def get_cell_code(cell: str) -> str:
    """Extract base station if merge=True."""
    if cell == '': return cell
    match = re.match(r"([a-zA-Z]+)", cell)
    return match.group(1)

def get_cell_code2(cell: str) -> str:
    """Extract base station if merge=True."""
    if cell == '': return cell
    match = re.match(r"([a-zA-Z]+)", cell)
    return match.group(1)[1:]

MERGE = {
    "no_merge": lambda x: x,
    "simple" : get_cell_code,
    "2g3g" : get_cell_code2
}
merge_cache = {name: {} for name in MERGE}


def merged(cells, name):
    cache = merge_cache[name]
    out = []
    for c in cells:
        if c not in cache:
            cache[c] = MERGE[name](c)
        out.append(cache[c])
    return out


def scan_file(path):
    """Un passage sur un fichier : compteurs du jour, histogramme de n_records, records par minute."""
    stats = Counter()
    n_records_hist = Counter()
    per_minute = np.zeros(MINUTES_PER_DAY, dtype=np.int64)
    cells_seen = set()
    stay_rates = {name: [] for name in MERGE}

    with open(path, newline="", encoding="utf-8") as f:
        for line in csv.reader(f, delimiter=";"):
            cells = line[8::2]
            timestamps = [int(ts) for ts in line[9::2]]
            n = len(cells)

            stats["user_days"] += 1
            stats["records"] += n
            stats["pairs"] += n - 1
            stats["same_cell_pairs"] += sum(a == b for a, b in zip(cells, cells[1:]))
            stats["bs1"] += line[5] != ""
            stats["bs2"] += line[6] != ""
            stats["bs1_and_bs2"] += line[5] != "" and line[6] != ""
            stats["bs1_equals_bs2"] += line[5] != "" and line[5] == line[6]
            stats["over_max_records"] += n > MAX_RECORDS

            n_records_hist[n] += 1
            cells_seen.update(cells)
            stamps = np.array(timestamps)
            np.add.at(per_minute, np.minimum(stamps // 60, MINUTES_PER_DAY - 1), 1)

            gaps = np.diff(stamps)
            stats["gaps_3h59_4h01"] += int(((gaps >= PERIODIC_UPDATE[0]) & (gaps < PERIODIC_UPDATE[1])).sum())
            for name, limit in GAP_LIMITS.items():
                over = int((gaps > limit).sum())
                stats[f"gaps_over_{name}"] += over
                stats[f"user_days_with_gap_over_{name}"] += over > 0

            if n >= MIN_RECORDS:
                for name in MERGE:
                    seq = merged(cells, name)
                    stay_rates[name].append(sum(a == b for a, b in zip(seq, seq[1:])) / (n - 1))

    return stats, n_records_hist, per_minute, cells_seen, stay_rates


def weekday_name(day):
    """'2014-03-12' -> 'Wed'."""
    return datetime.date.fromisoformat(day).strftime("%a")


def weighted_quantile(hist, q):
    """Quantile d'une distribution donnée sous forme {valeur: effectif}."""
    values = np.array(sorted(hist))
    cum = np.cumsum([hist[v] for v in values])
    return int(values[np.searchsorted(cum, q * cum[-1])])


def plot_records_hist(hist, out):
    """Fonction de répartition complémentaire : part des utilisateurs-jours ayant au moins n records."""
    values = np.array(sorted(hist))
    counts = np.array([hist[v] for v in values])
    total = counts.sum()
    median, mean = weighted_quantile(hist, 0.5), (values * counts).sum() / total
    share_over = counts[values > MAX_RECORDS].sum() / total
    at_least = counts[::-1].cumsum()[::-1] / total

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.step(values, at_least, where="post", color="#4C72B0", linewidth=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(median, color="black", linestyle="--", linewidth=1, label=f"median = {median} records")
    ax.axvline(mean, color="black", linestyle=":", linewidth=1, label=f"mean = {mean:.1f} records")
    ax.axvline(MAX_RECORDS, color="crimson", linewidth=1.2,
               label=f"{MAX_RECORDS}-record cut-off ({100 * share_over:.2f} % of user-days above)")
    ax.set_xlabel("Number of records n in the user-day (log scale)")
    ax.set_ylabel("Share of user-days with at least n records (log scale)")
    ax.set_title(f"Records per user-day: {total:,} user-days (largest: {values.max():,} records)".replace(",", " "))
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_minute_heatmap(per_minute_by_day, out):
    days = sorted(per_minute_by_day)
    grid = np.array([per_minute_by_day[d].reshape(-1, BIN_MINUTES).sum(axis=1) for d in days])

    fig, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", interpolation="nearest",
                   extent=(0, 24, len(days), 0))
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("Hour of the day")
    ax.set_yticks(np.arange(len(days)) + 0.5)
    ax.set_yticklabels([f"{d} {weekday_name(d)}" for d in days])
    for label, d in zip(ax.get_yticklabels(), days):
        if is_weekend(d):
            label.set_color("crimson")
    ax.set_title(f"Records per {BIN_MINUTES}-minute bin, one row per day (weekends in red)")
    fig.colorbar(im, ax=ax, label=f"Records per {BIN_MINUTES} min")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    rows = []
    total_hist = Counter()
    per_minute_by_day = {}
    all_cells = set()
    all_stay_rates = {name: [] for name in MERGE}

    for path in tqdm.tqdm(files, desc="Scanning days"):
        day = get_day(path)
        stats, hist, per_minute, cells_seen, stay_rates = scan_file(path)
        for name in MERGE:
            all_stay_rates[name].extend(stay_rates[name])
        total_hist.update(hist)
        per_minute_by_day[day] = per_minute
        all_cells |= cells_seen
        rows.append({"day": day, "weekday": weekday_name(day), "weekend": is_weekend(day),
                     "median_records": weighted_quantile(hist, 0.5), **stats})

    df = pd.DataFrame(rows)
    counters = [c for c in df.columns if c not in ("day", "weekday", "weekend", "median_records")]
    total = {"day": "TOTAL", "weekday": "", "weekend": "",
             "median_records": weighted_quantile(total_hist, 0.5), **df[counters].sum().to_dict()}
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)

    df["records_per_user"] = df["records"] / df["user_days"]
    df["same_cell_share"] = df["same_cell_pairs"] / df["pairs"]
    df["bs1_share"] = df["bs1"] / df["user_days"]
    df["bs2_share"] = df["bs2"] / df["user_days"]
    df.to_csv(INTERMEDIATE_PATH / "dataset_volume_by_day.csv", sep=";", index=False)

    plot_records_hist(total_hist, PLOT_PATH / "records_per_user_day_hist.png")
    plot_minute_heatmap(per_minute_by_day, PLOT_PATH / "records_per_minute_by_day.png")

    pd.set_option("display.width", 250)
    print(df[["day", "weekday", "user_days", "records", "records_per_user", "median_records",
              "same_cell_share", "bs1_share", "bs2_share"]].to_string(index=False))
    t = df.iloc[-1]
    print(f"\nDistinct cells: {len(all_cells)}")
    print(f"BS1 and BS2: {t.bs1_and_bs2}  BS1 only: {t.bs1 - t.bs1_and_bs2}  "
          f"BS2 only: {t.bs2 - t.bs1_and_bs2}  neither: {t.user_days - t.bs1 - t.bs2 + t.bs1_and_bs2}  "
          f"BS1 == BS2: {t.bs1_equals_bs2}")
    print(f"User-days over {MAX_RECORDS} records: {t.over_max_records} "
          f"({100 * t.over_max_records / t.user_days:.2f} %)")
    print(f"Largest user-day: {max(total_hist)} records")

    # ── Indicateurs globaux : gaps et fusions ──
    summary = {"pairs": t.pairs, "gaps_3h59_4h01": t.gaps_3h59_4h01}
    for name in GAP_LIMITS:
        summary[f"gaps_over_{name}"] = t[f"gaps_over_{name}"]
        summary[f"user_days_with_gap_over_{name}"] = t[f"user_days_with_gap_over_{name}"]
    technologies = {}
    for cell in all_cells:
        technologies.setdefault(get_cell_code2(cell), set()).add("2G" if cell[0] in "BD" else "3G")
    summary["base_stations"] = len(technologies)
    summary["base_stations_2g_and_3g"] = sum(len(v) == 2 for v in technologies.values())
    summary["base_stations_2g_only"] = sum(v == {"2G"} for v in technologies.values())
    summary["base_stations_3g_only"] = sum(v == {"3G"} for v in technologies.values())
    summary["cells_on_2g_and_3g_base_stations"] = sum(len(technologies[get_cell_code2(c)]) == 2 for c in all_cells)
    summary[f"user_days_min_{MIN_RECORDS}_records"] = len(all_stay_rates["no_merge"])
    for name in MERGE:
        summary[f"symbols_{name}"] = len({MERGE[name](c) for c in all_cells})
        summary[f"stay_rate_mean_{name}"] = round(float(np.mean(all_stay_rates[name])), 4)
        summary[f"stay_rate_median_{name}"] = round(float(np.median(all_stay_rates[name])), 4)
    pd.Series(summary, name="value").rename_axis("indicator").to_csv(
        INTERMEDIATE_PATH / "dataset_volume_summary.csv", sep=";")
    print("\nGaps and merges:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    for q in (0.25, 0.5, 0.75, 0.9, 0.99, 0.994):
        print(f"  quantile {q}: {weighted_quantile(total_hist, q)} records")
