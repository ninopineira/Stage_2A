import numpy as np
import json
from pathlib import Path

MAIN_DIR   = Path(__file__).parent.parent.parent
NUMPY_DIR  = MAIN_DIR / "results/numpy"
OUTPUT_DIR = MAIN_DIR / "results/plots"

MERGES         = ["no_merge", "simple", "2g3g"]
MERGE_LABELS   = ["No merge", "Simple", "2G / 3G"]
PERIOD_NAMES   = ["Morning", "Day", "Evening"]
MORNING_ENDS   = [4, 5, 6]
EVENING_STARTS = [18, 19, 20]
N_BINS         = 50

# ─────────────────────────────────────────────────────────────────────────────
# P_MAX  via vectorised bisection of the Fano inequality
#   H(p) + (1-p)·log2(N-1) = S_unc   →  solve for p = P_max
# ─────────────────────────────────────────────────────────────────────────────
def pmax_vec(s_unc, n_eff, n_iter=60):
    result = np.full(len(s_unc), np.nan)
    valid  = (n_eff > 1) & (s_unc > 0)
    s, n   = s_unc[valid], n_eff[valid]

    uniform   = s >= np.log2(n)          # max entropy → uniform
    result_v  = np.where(uniform, 1.0 / n, np.nan)

    nonu = ~uniform
    if nonu.any():
        sv, nv = s[nonu], n[nonu]
        lo = np.full(nonu.sum(), 1e-10)
        hi = np.ones(nonu.sum()) - 1e-10
        for _ in range(n_iter):
            mid = (lo + hi) / 2
            lm, l1m = np.log2(np.maximum(mid, 1e-300)), np.log2(np.maximum(1 - mid, 1e-300))
            h   = -mid * lm - (1 - mid) * l1m
            f   = h + (1 - mid) * np.log2(np.maximum(nv - 1, 1)) - sv
            lo  = np.where(f > 0, mid, lo)
            hi  = np.where(f < 0, mid, hi)
        result_v[nonu] = (lo + hi) / 2

    result[valid] = result_v
    return result

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("Loading npy files...")
raw_all = {}
for merge in MERGES:
    raw_all[merge] = np.load(
        NUMPY_DIR / f"user_entropies_by_period_{merge}.npy", allow_pickle=True
    ).item()

# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — global max of S_unc and S_rand per (p,mi,ei) for consistent axes
#           S_rel and P_max are always in [0,1] → fixed edges
# ─────────────────────────────────────────────────────────────────────────────
print("Pass 1: computing ranges...")
max_unc  = np.zeros((3, 3, 3))
max_rand = np.zeros((3, 3, 3))

for merge in MERGES:
    for day, users in raw_all[merge].items():
        arr = np.array(users, dtype=float)
        for p in range(3):
            for mi in range(3):
                for ei in range(3):
                    ci    = mi * 3 + ei
                    s_unc = arr[:, ci * 3 + p]
                    s_rel = arr[:, ci * 3 + p + 27]
                    valid = (s_unc > 0) & (s_rel > 0)
                    if not valid.any():
                        continue
                    unc_v  = s_unc[valid]
                    rand_v = np.log2(np.maximum(unc_v / s_rel[valid], 1.0))
                    max_unc[p, mi, ei]  = max(max_unc[p, mi, ei],  float(unc_v.max()))
                    max_rand[p, mi, ei] = max(max_rand[p, mi, ei], float(rand_v.max()))

edges_unc  = [[[np.linspace(0, max_unc[p,mi,ei]  or 1.0, N_BINS+1) for ei in range(3)] for mi in range(3)] for p in range(3)]
edges_rand = [[[np.linspace(0, max_rand[p,mi,ei] or 1.0, N_BINS+1) for ei in range(3)] for mi in range(3)] for p in range(3)]
edges_rel  = np.linspace(0, 1, N_BINS + 1)   # S_rel ∈ [0,1]
edges_pmax = np.linspace(0, 1, N_BINS + 1)   # P_max ∈ [0,1]

# ─────────────────────────────────────────────────────────────────────────────
# PASS 2 — accumulate counts for all four metrics
# ─────────────────────────────────────────────────────────────────────────────
print("Pass 2: computing histograms...")
shape = (3, 3, 3, N_BINS)

cnt_unc  = {m: np.zeros(shape) for m in MERGES}
cnt_rand = {m: np.zeros(shape) for m in MERGES}
cnt_rel  = {m: np.zeros(shape) for m in MERGES}
cnt_pmax = {m: np.zeros(shape) for m in MERGES}

sum_unc  = {m: np.zeros((3,3,3)) for m in MERGES}
sum_rand = {m: np.zeros((3,3,3)) for m in MERGES}
sum_rel  = {m: np.zeros((3,3,3)) for m in MERGES}
sum_pmax = {m: np.zeros((3,3,3)) for m in MERGES}
n_pts    = {m: np.zeros((3,3,3)) for m in MERGES}

for merge in MERGES:
    print(f"  {merge}...")
    for day, users in raw_all[merge].items():
        arr = np.array(users, dtype=float)
        for p in range(3):
            for mi in range(3):
                for ei in range(3):
                    ci    = mi * 3 + ei
                    s_unc = arr[:, ci * 3 + p]
                    s_rel = arr[:, ci * 3 + p + 27]
                    valid = (s_unc > 0) & (s_rel > 0)
                    if not valid.any():
                        continue

                    unc_v  = s_unc[valid]
                    rel_v  = s_rel[valid]
                    n_eff  = unc_v / rel_v
                    rand_v = np.log2(np.maximum(n_eff, 1.0))
                    pmax_v = pmax_vec(unc_v, n_eff)
                    ok     = ~np.isnan(pmax_v)

                    cu, _ = np.histogram(unc_v,        bins=edges_unc[p][mi][ei])
                    cr, _ = np.histogram(rand_v,       bins=edges_rand[p][mi][ei])
                    cs, _ = np.histogram(rel_v,        bins=edges_rel)
                    cp, _ = np.histogram(pmax_v[ok],   bins=edges_pmax)

                    cnt_unc[merge][p,mi,ei]  += cu
                    cnt_rand[merge][p,mi,ei] += cr
                    cnt_rel[merge][p,mi,ei]  += cs
                    cnt_pmax[merge][p,mi,ei] += cp

                    sum_unc[merge][p,mi,ei]  += float(unc_v.sum())
                    sum_rand[merge][p,mi,ei] += float(rand_v.sum())
                    sum_rel[merge][p,mi,ei]  += float(rel_v.sum())
                    sum_pmax[merge][p,mi,ei] += float(pmax_v[ok].sum())
                    n_pts[merge][p,mi,ei]    += valid.sum()

# ─────────────────────────────────────────────────────────────────────────────
# BUILD JSON PAYLOAD
# ─────────────────────────────────────────────────────────────────────────────
def to_density(counts, bin_width):
    total = float(counts.sum())
    if total == 0 or bin_width == 0:
        return counts.round(4).tolist()
    return (counts / (total * bin_width)).round(4).tolist()

cx_rel  = ((edges_rel[:-1]  + edges_rel[1:])  / 2).round(4).tolist()
cx_pmax = ((edges_pmax[:-1] + edges_pmax[1:]) / 2).round(4).tolist()
w_rel   = float(edges_rel[1]  - edges_rel[0])
w_pmax  = float(edges_pmax[1] - edges_pmax[0])

hist = {}
for p in range(3):
    for mi in range(3):
        for ei in range(3):
            k  = f"{p}_{mi}_{ei}"
            eu = edges_unc[p][mi][ei]
            er = edges_rand[p][mi][ei]
            cx_u = ((eu[:-1] + eu[1:]) / 2).round(4).tolist()
            cx_r = ((er[:-1] + er[1:]) / 2).round(4).tolist()
            wu, wr = float(eu[1]-eu[0]), float(er[1]-er[0])
            hist[k] = {}
            for merge in MERGES:
                n = float(n_pts[merge][p,mi,ei])
                def mean(s): return round(float(s[merge][p,mi,ei] / n), 4) if n > 0 else 0.0
                hist[k][merge] = {
                    "su_x":   cx_u,
                    "su_y":   to_density(cnt_unc[merge][p,mi,ei],  wu),
                    "sr_x":   cx_r,
                    "sr_y":   to_density(cnt_rand[merge][p,mi,ei], wr),
                    "ss_x":   cx_rel,
                    "ss_y":   to_density(cnt_rel[merge][p,mi,ei],  w_rel),
                    "sp_x":   cx_pmax,
                    "sp_y":   to_density(cnt_pmax[merge][p,mi,ei], w_pmax),
                    "mu":  mean(sum_unc),
                    "mr":  mean(sum_rand),
                    "ms":  mean(sum_rel),
                    "mp":  mean(sum_pmax),
                }

payload = {
    "hist": hist,
    "merges":        MERGES,
    "mergeLabels":   MERGE_LABELS,
    "periodNames":   PERIOD_NAMES,
    "morningEnds":   MORNING_ENDS,
    "eveningStarts": EVENING_STARTS,
}
data_json = json.dumps(payload, separators=(",", ":"))

# ─────────────────────────────────────────────────────────────────────────────
# HTML  — 3 rows (merge) × 3 cols (Srand+Sunc | Srel | Pmax)
# ─────────────────────────────────────────────────────────────────────────────
COL_HEADER_HTML = (
    '      <div class="col-headers">\n'
    '        <div class="col-header">P(S<sup>rand</sup>) &amp; P(S<sup>unc</sup>)</div>\n'
    '        <div class="col-header">P(S<sup>rel</sup>)&nbsp;&nbsp;relative entropy</div>\n'
    '        <div class="col-header">P(P<sup>max</sup>)&nbsp;&nbsp;predictability</div>\n'
    '      </div>\n'
)

merge_sections = "\n".join(
    '    <div class="merge-section">\n'
    f'      <div class="row-header" id="row-header-{row}"></div>\n' +
    COL_HEADER_HTML +
    '      <div class="charts-row">\n' +
    "\n".join(
        f'        <div id="chart-{row}-{col}" class="chart-cell"></div>'
        for col in range(3)
    ) +
    '\n      </div>\n    </div>'
    for row in range(len(MERGES))
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Entropy histograms by period</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; background: #f0f2f5; padding: 16px; }}
    h1 {{ text-align: center; margin-bottom: 14px; font-size: 1.15em; color: #333; }}
    .controls {{
      display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;
      background: white; padding: 12px 24px; border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,.12); margin-bottom: 12px;
    }}
    .control-group {{ display: flex; flex-direction: column; gap: 4px; }}
    .control-group label {{
      font-size: .72em; color: #555; font-weight: bold;
      text-transform: uppercase; letter-spacing: .04em;
    }}
    select {{
      padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px;
      font-size: .88em; background: white; cursor: pointer; outline: none;
    }}
    select:focus {{ border-color: #4C72B0; box-shadow: 0 0 0 2px rgba(76,114,176,.2); }}
    .selection-display {{
      text-align: center; font-size: 1em; font-weight: bold;
      color: #333; margin-bottom: 14px;
    }}
    .col-headers {{
      display: grid; grid-template-columns: repeat(3, 1fr);
      gap: 12px; margin-bottom: 6px;
    }}
    .col-header {{
      text-align: center; font-weight: bold; font-size: .85em;
      color: #444; padding: 5px; background: white;
      border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,.07);
    }}
    .merge-section {{ margin-bottom: 20px; }}
    .row-header {{
      font-weight: bold; font-size: 1.05em; color: #333;
      padding: 6px 4px 6px 4px;
    }}
    .charts-row {{
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
    }}
    .chart-cell {{
      background: white; border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,.08);
      height: 320px;
    }}
  </style>
</head>
<body>
  <h1>Entropy and P<sup>max</sup> distributions by time window</h1>
  <div class="controls">
    <div class="control-group">
      <label>Period</label>
      <select id="period">
        <option value="0">Morning</option>
        <option value="1" selected>Day</option>
        <option value="2">Evening</option>
      </select>
    </div>
    <div class="control-group">
      <label>Morning end</label>
      <select id="morning-end">
        <option value="0">4h</option>
        <option value="1">5h</option>
        <option value="2">6h</option>
      </select>
    </div>
    <div class="control-group">
      <label>Evening start</label>
      <select id="evening-start">
        <option value="0">18h</option>
        <option value="1">19h</option>
        <option value="2">20h</option>
      </select>
    </div>
  </div>
  <div class="selection-display" id="selection-display"></div>
{merge_sections}

  <script>
    const P = {data_json};

    const COL_CONFIGS = [
      {{
        traces: (d) => [
          {{ x: d.sr_x, y: d.sr_y, type:'bar', name:'Sʳᵃⁿᵈ=log₂(N)',
             marker:{{color:'gold', opacity:0.75}}, width: d.sr_x[1]-d.sr_x[0] }},
          {{ x: d.su_x, y: d.su_y, type:'bar', name:'Sᵘⁿᶜ (Shannon)',
             marker:{{color:'mediumseagreen', opacity:0.75}}, width: d.su_x[1]-d.su_x[0] }},
          {{ x:[d.mr,d.mr], y:[0,Math.max(...d.sr_y)], type:'scatter', mode:'lines',
             name:'Mean Sʳᵃⁿᵈ='+d.mr.toFixed(2), line:{{color:'goldenrod',dash:'dash',width:2}} }},
          {{ x:[d.mu,d.mu], y:[0,Math.max(...d.su_y)], type:'scatter', mode:'lines',
             name:'Mean Sᵘⁿᶜ='+d.mu.toFixed(2),  line:{{color:'seagreen', dash:'dash',width:2}} }}
        ],
        xLabel: 'Entropy (bits)'
      }},
      {{
        traces: (d) => [
          {{ x: d.ss_x, y: d.ss_y, type:'bar', name:'Sʳᵉˡ',
             marker:{{color:'steelblue', opacity:0.75}}, width: d.ss_x[1]-d.ss_x[0] }},
          {{ x:[d.ms,d.ms], y:[0,Math.max(...d.ss_y)], type:'scatter', mode:'lines',
             name:'Mean='+d.ms.toFixed(3), line:{{color:'navy',dash:'dash',width:2}} }}
        ],
        xLabel: 'Relative entropy'
      }},
      {{
        traces: (d) => [
          {{ x: d.sp_x, y: d.sp_y, type:'bar', name:'Pᵐᵃˣ',
             marker:{{color:'mediumpurple', opacity:0.75}}, width: d.sp_x[1]-d.sp_x[0] }},
          {{ x:[d.mp,d.mp], y:[0,Math.max(...d.sp_y)], type:'scatter', mode:'lines',
             name:'Mean='+d.mp.toFixed(3), line:{{color:'purple',dash:'dash',width:2}} }}
        ],
        xLabel: 'P max'
      }}
    ];

    function draw() {{
      const period = document.getElementById('period').value;
      const mEnd   = document.getElementById('morning-end').value;
      const eStart = document.getElementById('evening-start').value;
      const key    = period + '_' + mEnd + '_' + eStart;
      const pLabel = P.periodNames[+period];
      const mLabel = P.morningEnds[+mEnd];
      const eLabel = P.eveningStarts[+eStart];

      document.getElementById('selection-display').textContent =
        pLabel + '  |  morning → ' + mLabel + 'h  |  evening ← ' + eLabel + 'h';

      P.merges.forEach((merge, row) => {{
        document.getElementById('row-header-' + row).textContent = P.mergeLabels[row];
        const d = P.hist[key][merge];
        COL_CONFIGS.forEach((cfg, col) => {{
          Plotly.react('chart-' + row + '-' + col,
            cfg.traces(d),
            {{
              barmode: 'overlay',
              margin: {{ t: 15, r: 15, b: 45, l: 50 }},
              xaxis: {{ title: {{text: cfg.xLabel, font:{{size:9}}}}, tickfont:{{size:8}} }},
              yaxis: {{ title: {{text: 'Density',  font:{{size:9}}}}, tickfont:{{size:8}} }},
              legend: {{ font:{{size:8}}, x:0.98, xanchor:'right', y:0.98 }},
              paper_bgcolor:'white', plot_bgcolor:'#fafafa'
            }},
            {{responsive: true}}
          );
        }});
      }});
    }}

    document.querySelectorAll('select').forEach(s => s.addEventListener('change', draw));
    draw();
  </script>
</body>
</html>"""

output_path = OUTPUT_DIR / "entropy_hist_by_period.html"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Saved: {output_path}")
