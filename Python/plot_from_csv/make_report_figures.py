# Gathers every figure used by the two reports (rapport/rapport_stage_2A.tex and
# rapport/report_detailed_en.tex) into rapport/figures/, from their source in results/.
# Single source of truth for the "figure -> source" mapping given in the appendices:
# re-run this script after regenerating any result, then recompile the reports.
#
#   - plain copies for most figures;
#   - two composites (activity-cell comparison) assembled from the plots of
#     important_cells_work/2_activity_cells_results_against_dataset.py;
#   - a headless-Chrome screenshot of the folium entry/exit map.
#
# Figures not listed here (antenne_kvare_*.png) were made by hand and are left untouched.

import shutil
import subprocess
from pathlib import Path

from PIL import Image

MAIN_DIR = Path(__file__).parent.parent.parent
RESULTS = MAIN_DIR / "results"
FIG_DIR = MAIN_DIR / "rapport/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SP = "predictions/simple_predictor/plots"
MV = "predictions/movement_prediction_V2/plots_V1"
ACT = "intermediate_result/activity_cells_comparison/plots"

COPIES = {
    "occupation_horaire.png": "plots/stats_connections_cell_by_hour_by_day.png",
    "presence_par_jour.png": "plots/sum_by_day_user_presence.png",
    "classification_utilisateurs.png": "plots/classification_by_presence_5_clusters.png",
    "antennes_centroides.png": "2014-03-12/cluster_centroids_heatmap.png",
    "song_replication.png": "plots/song_replication_no_merge.png",
    "entropie_par_periode.png": "plots/entropy_by_period_no_merge.png",
    "markov_methods_by_day.png": "plots/markov_methods_by_day.png",
    "markov_vs_pmax_calendar.png": "plots/markov_sequential_vs_pmax_calendar.png",
    "markov_vs_vomm.png": f"{SP}/markov_vs_vomm_no_duplicate.png",
    "vomm_discount.png": f"{SP}/vomm_discount_no_duplicate.png",
    "pmax_unc.png": f"{SP}/predictability_vs_pmax_unc.png",
    "pmax_cond.png": f"{SP}/predictability_vs_pmax_cond.png",
    "predictability_by_band.png": f"{SP}/predictability_by_band.png",
    "heldout_demo.png": f"{SP}/predictability_heldout_demonstration_calendar.png",
    "heldout_weekday.png": f"{SP}/predictability_heldout_parallel_weekday.png",
    "lz_weekday.png": f"{SP}/predictability_lz_parallel_weekday.png",
    "movestay_roc_pr.png": f"{MV}/roc_pr.png",
    "movestay_importance.png": f"{MV}/feature_importance.png",
    "movestay_threshold.png": f"{MV}/threshold_analysis.png",
    "movestay_entropy_ablation.png": f"{MV}/movement_entropy_before_after.png",
}

COMPOSITES = {
    # name: (layout, [sources])
    "activity_algos_comparaison.png": ("vertical", [f"{ACT}/01_match_rate_by_day.png",
                                                    f"{ACT}/03_coverage_by_day.png"]),
    "activity_algos_agreement.png": ("horizontal", [f"{ACT}/04_cross_agreement_heatmap.png",
                                                    f"{ACT}/05_breakdown_per_day_per_method.png"]),
}

MAP_HTML = RESULTS / "maps/entrance_exit_no_merge.html"
MAP_PNG = "carte_entrees_sorties.png"
CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")


def compose(layout, paths, out):
    """Stack images, scaling them to a common width (vertical) or height (horizontal)."""
    imgs = [Image.open(p).convert("RGB") for p in paths]
    if layout == "vertical":
        w = max(i.width for i in imgs)
        imgs = [i.resize((w, round(i.height * w / i.width))) for i in imgs]
        canvas = Image.new("RGB", (w, sum(i.height for i in imgs)), "white")
        y = 0
        for i in imgs:
            canvas.paste(i, (0, y))
            y += i.height
    else:
        h = max(i.height for i in imgs)
        imgs = [i.resize((round(i.width * h / i.height), h)) for i in imgs]
        canvas = Image.new("RGB", (sum(i.width for i in imgs), h), "white")
        x = 0
        for i in imgs:
            canvas.paste(i, (x, 0))
            x += i.width
    canvas.save(out)


def screenshot_map(out):
    if not CHROME.exists():
        print(f"  !! Chrome not found, map screenshot skipped ({CHROME})")
        return
    subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--window-size=1600,1100", "--virtual-time-budget=15000",
                    f"--screenshot={out}", MAP_HTML.as_uri()], check=True, capture_output=True)


if __name__ == "__main__":
    missing = []
    for name, src in COPIES.items():
        p = RESULTS / src
        if p.exists():
            shutil.copy2(p, FIG_DIR / name)
            print(f"  copied    {name:34s} <- {src}")
        else:
            missing.append(src)
    for name, (layout, srcs) in COMPOSITES.items():
        paths = [RESULTS / s for s in srcs]
        if all(p.exists() for p in paths):
            compose(layout, paths, FIG_DIR / name)
            print(f"  composed  {name:34s} <- {', '.join(Path(s).name for s in srcs)}")
        else:
            missing += [s for s, p in zip(srcs, paths) if not p.exists()]
    if MAP_HTML.exists():
        screenshot_map(FIG_DIR / MAP_PNG)
        print(f"  captured  {MAP_PNG:34s} <- {MAP_HTML.relative_to(RESULTS)}")
    else:
        missing.append(str(MAP_HTML.relative_to(RESULTS)))
    if missing:
        print("\nMissing sources (run the producing scripts first):")
        for m in missing:
            print("  -", m)
