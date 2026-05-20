from pathlib import Path
import pandas as pd
import json

MAIN_DIR = Path(__file__).parent.parent.parent
STATS_DIR = MAIN_DIR / "results/intermediate_result"
OUTPUT_DIR = MAIN_DIR / "results/plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MERGE_FILES = {
    "No merge": "stats_use_cell_by_hour.csv",
    "Simple merge": "stats_use_cell_by_hour_simple_merge.csv",
    "2G/3G merge": "stats_use_cell_by_hour_2g3g_merge.csv",
}

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <style>
    body {{
      font-family: Arial, sans-serif;
      max-width: 960px;
      margin: 40px auto;
      padding: 0 20px;
      background: #f5f5f5;
    }}
    h1 {{ color: #333; }}
    label {{ font-size: 1rem; margin-right: 8px; }}
    select {{
      font-size: 1rem;
      padding: 4px 8px;
      border-radius: 4px;
      border: 1px solid #aaa;
    }}
    .chart-container {{
      background: #fff;
      border-radius: 8px;
      padding: 20px;
      margin-top: 24px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <label for="station-select">Station :</label>
  <select id="station-select"></select>

  <div class="chart-container">
    <canvas id="bar-chart"></canvas>
  </div>

  <script>
    const DATA   = {data_json};
    const LABELS = {labels_json};

    const select = document.getElementById("station-select");
    const ctx    = document.getElementById("bar-chart").getContext("2d");

    // Populate dropdown
    Object.keys(DATA).forEach(station => {{
      const opt = document.createElement("option");
      opt.value = station;
      opt.textContent = station;
      select.appendChild(opt);
    }});

    const chart = new Chart(ctx, {{
      type: "bar",
      data: {{
        labels: LABELS,
        datasets: [{{
          label: "Number of connections",
          data: [],
          backgroundColor: "rgba(54, 162, 235, 0.7)",
          borderColor: "rgba(54, 162, 235, 1)",
          borderWidth: 1
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ display: false }},
          title: {{
            display: true,
            text: "",
            font: {{ size: 15 }}
          }}
        }},
        scales: {{
          x: {{ title: {{ display: true, text: "Hour of the day" }} }},
          y: {{ title: {{ display: true, text: "Number of connections" }}, beginAtZero: true }}
        }}
      }}
    }});

    function updateChart(station) {{
      chart.data.datasets[0].data = DATA[station];
      chart.options.plugins.title.text = `Station : ${{station}}`;
      chart.update();
    }}

    select.addEventListener("change", () => updateChart(select.value));

    // Draw first station on load
    if (Object.keys(DATA).length > 0) {{
      updateChart(Object.keys(DATA)[0]);
    }}
  </script>
</body>
</html>
"""


def df_to_json_data(df: pd.DataFrame) -> dict:
    """Convert DataFrame (rows=stations, cols=hours) to {station: [values]} dict."""
    return {str(idx): row.tolist() for idx, row in df.iterrows()}


for merge_name, filename in MERGE_FILES.items():
    csv_path = STATS_DIR / filename
    if not csv_path.exists():
        print(f"File not found, skipping: {csv_path}")
        continue

    df = pd.read_csv(csv_path, sep=";", index_col=0)
    hour_labels = df.columns.tolist()

    data = df_to_json_data(df)
    title = f"Cell use by hour — {merge_name}"

    html = HTML_TEMPLATE.format(
        title=title,
        data_json=json.dumps(data, ensure_ascii=False),
        labels_json=json.dumps(hour_labels, ensure_ascii=False),
    )

    stem = filename.replace(".csv", "")
    out_path = OUTPUT_DIR / f"{stem}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Generated: {out_path}")
