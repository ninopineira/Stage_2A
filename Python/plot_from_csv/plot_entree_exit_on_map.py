import folium
import json
from pathlib import Path
import pandas as pd
import re

MAIN_DIR = Path(__file__).parent.parent.parent
CELLS_FILE = MAIN_DIR / "Database/cells/cd_142_cells.csv"
STATS_DIR = MAIN_DIR / "results/intermediate_result"
OUTPUT_DIR = MAIN_DIR / "results/maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def name_cluster(cluster_code : int):
    if not 0 <= cluster_code <= 4:
        return "Unknown"
    
    cluster_names = {
        0 : "Always present",
        1 : "Home but out workers",
        2 : "Night residents leaving during the day",
        3 : "Day arrivals staying overnight",
        4 : "Passing through the area"
    }

    return cluster_names.get(cluster_code, "Unknown")

def extract_letters(cellid: str) -> str:
    match = re.match(r"([a-zA-Z]+)", cellid)
    return match.group(1) if match else cellid


if not CELLS_FILE.exists():
    print(f"Cell database not found: {CELLS_FILE}")
    exit(1)

cells = pd.read_csv(CELLS_FILE, sep=";")
cells["key_no_merge"] = cells["cellid"]
cells["key_simple"]   = cells["cellid"].apply(extract_letters)
cells["key_2g3g"]     = cells["cellid"].apply(lambda c: extract_letters(c)[1:])


def station_coords(key_col: str) -> pd.DataFrame:
    return (
        cells.groupby(key_col)[["lat", "lon"]]
        .mean()
        .reset_index()
        .rename(columns={key_col: "station"})
    )


CONFIGS = [
    ("stats_entrance_exit_no_merge.csv",     station_coords("key_no_merge"), "entrance_exit_no_merge.html",     "No merge"),
    ("stats_entrance_exit_simple_merge.csv", station_coords("key_simple"),   "entrance_exit_simple_merge.html", "Simple merge"),
    ("stats_entrance_exit_2g3g_merge.csv",   station_coords("key_2g3g"),     "entrance_exit_2g3g_merge.html",   "2G/3G merge"),
]

for stats_file, coords, output_file, title in CONFIGS:
    stats_path = STATS_DIR / stats_file
    if not stats_path.exists():
        print(f"[{title}] fichier manquant : {stats_path}")
        continue

    stats = pd.read_csv(stats_path, sep=";")
    stats = stats.dropna(subset=["cluster_type"])
    stats["cluster_type"] = stats["cluster_type"].astype(int)
    clusters = sorted(set(range(5)) | set(stats["cluster_type"].unique().tolist()))

    # Aggregate by (station, cluster_type) summing over all days
    agg = stats.groupby(["station", "cluster_type"])[["entrance_count", "exit_count"]].sum().reset_index()
    merged = agg.merge(coords, on="station", how="inner")
    if merged.empty:
        print(f"[{title}] aucune station avec coordonnées connues")
        continue

    center_lat = (merged["lat"].max() + merged["lat"].min()) / 2
    center_lon = (merged["lon"].max() + merged["lon"].min()) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    map_var = m.get_name()

    # Stations layer — one dot per station, cluster-agnostic
    fg_stations = folium.FeatureGroup(name="Stations", show=True).add_to(m)
    for station, grp in merged.groupby("station"):
        row = grp.iloc[0]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=1,
            color="black",
            fill=True,
            fill_color="black",
            fill_opacity=1.0,
            tooltip=station,
        ).add_to(fg_stations)

    # Entrance / exit layers start empty — JS fills them based on cluster selection
    fg_in  = folium.FeatureGroup(name="Entrées",  show=True).add_to(m)
    fg_out = folium.FeatureGroup(name="Sorties", show=True).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Serialize per-station per-cluster data for JS
    station_data: dict = {}
    for _, row in merged.iterrows():
        s = row["station"]
        c = int(row["cluster_type"])
        if s not in station_data:
            station_data[s] = {"lat": row["lat"], "lon": row["lon"], "clusters": {}}
        station_data[s]["clusters"][c] = {
            "in":  int(row["entrance_count"]),
            "out": int(row["exit_count"]),
        }

    # Cluster checkboxes HTML
    cluster_checkboxes = ""
    for c in clusters:
        cluster_checkboxes += f"""
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer;margin:3px 0;">
          <input type="checkbox" id="cb-cluster-{c}" checked onchange="updateLayers()">
          {name_cluster(c)}
        </label>"""

    panel_html = f"""
    <div id="cluster-control" style="
        position: fixed;
        bottom: 30px;
        right: 10px;
        z-index: 1000;
        background: white;
        padding: 12px 16px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        font-size: 13px;
        min-width: 150px;
        line-height: 1.5;
    ">
      <div style="font-weight:bold;font-size:14px;margin-bottom:8px;border-bottom:1px solid #ddd;padding-bottom:5px;">
        Clusters
      </div>
      <div style="margin-bottom:6px;">
        <span style="cursor:pointer;color:#1a73e8;font-size:12px;" onclick="setAllClusters(true)">Tout</span>
        &nbsp;/&nbsp;
        <span style="cursor:pointer;color:#1a73e8;font-size:12px;" onclick="setAllClusters(false)">Aucun</span>
      </div>
      {cluster_checkboxes}
    </div>

    <script>
    window.addEventListener('load', function() {{
      var stationData = {json.dumps(station_data)};
      var allClusters = {json.dumps(clusters)};
      var fgIn        = {fg_in.get_name()};
      var fgOut       = {fg_out.get_name()};

      var MIN_R = 3, MAX_R = 20;

      function scaleRadius(val, maxVal) {{
        if (val === 0 || maxVal === 0) return 0;
        return MIN_R + (val / maxVal) * (MAX_R - MIN_R);
      }}

      window.updateLayers = function() {{
        fgIn.clearLayers();
        fgOut.clearLayers();

        var selected = allClusters.filter(function(c) {{
          var el = document.getElementById('cb-cluster-' + c);
          return el && el.checked;
        }});

        // First pass: compute totals and find the max for normalization
        var totals = {{}};
        var maxIn = 0, maxOut = 0;
        Object.entries(stationData).forEach(function([station, data]) {{
          var totalIn = 0, totalOut = 0;
          selected.forEach(function(c) {{
            if (data.clusters[c]) {{
              totalIn  += data.clusters[c]['in'];
              totalOut += data.clusters[c]['out'];
            }}
          }});
          totals[station] = {{in: totalIn, out: totalOut}};
          if (totalIn  > maxIn)  maxIn  = totalIn;
          if (totalOut > maxOut) maxOut = totalOut;
        }});

        // Second pass: draw circles with normalized radius
        Object.entries(totals).forEach(function([station, t]) {{
          var data = stationData[station];
          if (t.in > 0) {{
            L.circleMarker([data.lat, data.lon], {{
              radius: scaleRadius(t.in, maxIn),
              color: 'steelblue',
              fill: true,
              fillOpacity: 0.6,
            }}).bindTooltip(station + ' — entrées : ' + t.in).addTo(fgIn);
          }}
          if (t.out > 0) {{
            L.circleMarker([data.lat, data.lon], {{
              radius: scaleRadius(t.out, maxOut),
              color: 'orange',
              fill: true,
              fillOpacity: 0.6,
            }}).bindTooltip(station + ' — sorties : ' + t.out).addTo(fgOut);
          }}
        }});
      }};

      window.setAllClusters = function(state) {{
        allClusters.forEach(function(c) {{
          var el = document.getElementById('cb-cluster-' + c);
          if (el) el.checked = state;
        }});
        updateLayers();
      }};

      updateLayers();
    }});
    </script>
    """

    m.get_root().html.add_child(folium.Element(panel_html))
    m.save(OUTPUT_DIR / output_file)
    print(f"[{title}] {merged['station'].nunique()} stations, clusters {clusters} → {OUTPUT_DIR / output_file}")
