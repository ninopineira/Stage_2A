import numpy as np
import datetime
from pathlib import Path
import csv

# Source - https://stackoverflow.com/a/30609050
# Posted by Franck Dernoncourt, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-07, License - CC BY-SA 3.0
def find_ngrams(input_list : list[str], n : int):
    """
    finds all n-gramms in a list of strings
    input_list : a list of strings (ex : ['this', 'is', 'a', 'list'])
    n : the n-grams you want
    
    example : 
    
    inputs  : ['this', 'is', 'a', 'list'], 2
    
    outputs : iterator of the wanted n-grams, get all n-grams by converting it into a list :
    list(find_ngrams(inputs))
    """
    return zip(*[input_list[i:] for i in range(n)])

def find_ngrams_optimized(input_list, n, timestamps=None, max_gap=None):
    """
    Version vectorisée très performante.

    input_list : list[str] ou array-like
    n : taille du n-gram
    timestamps : list[int] ou None
    max_gap : int ou None

    Retour :
        array de shape (num_ngrams_valid, n)
    """

    x = np.asarray(input_list)
    L = x.shape[0]

    if L < n:
        return np.empty((0, n), dtype=x.dtype)

    # fenêtres glissantes sur les ids
    ngrams = np.lib.stride_tricks.sliding_window_view(x, n)

    if timestamps is None:
        return ngrams

    t = np.asarray(timestamps)

    if t.shape[0] != L:
        raise ValueError("timestamps must match input_list length")
    if max_gap is None:
        raise ValueError("max_gap must be provided")

    # calcul des gaps
    gaps = np.diff(t)  # shape (L-1,)

    # fenêtres de gaps (taille n-1)
    gap_windows = np.lib.stride_tricks.sliding_window_view(gaps, n-1)

    # masque : tous les gaps <= max_gap
    valid_mask = np.all(gap_windows <= max_gap, axis=1)
    
    return ngrams[valid_mask]

def find_ngrams_with_timestamps(stations: list[str], timestamps: list[int], n: int):
    """
    Retourne les n-grams de stations avec leurs timestamps correspondants.
    
    Retour : liste de (seq_str, timestamps_list) où :
        - seq_str        : "Cell_A-Cell_B-Cell_C"
        - timestamps_list: [ts_A, ts_B, ts_C]
    
    Note : les doublons de séquences sont conservés (contrairement à set()),
    car deux occurrences du même n-gram peuvent avoir des timestamps différents.
    Si tu veux les dédupliquer, utilise find_ngrams_with_timestamps_unique().
    """
    L = len(stations)
    if L < n:
        return []
    
    result = []
    for i in range(L - n + 1):
        seq_str = "-".join(stations[i:i+n])
        ts_list = timestamps[i:i+n]
        result.append((seq_str, ts_list))
    
    return result

def find_ngrams_with_timestamps_unique_fast(stations: list[str], timestamps: list[int], n: int):
    """
    Comme find_ngrams_with_timestamps mais déduplique les séquences identiques.
    Pour chaque séquence unique, ne garde qu'une occurrence (la première trouvée).
    C'est l'équivalent exact de ton set(find_ngrams(...)) mais avec les timestamps.
    """
    L = len(stations)
    if L < n:
        return []
    
    seen   = set()
    result = []
    sep    = "-"
    for i in range(L - n + 1):
        seq_str = sep.join(stations[i:i+n])
        if seq_str in seen:
            continue
        seen.add(seq_str)
        result.append((seq_str, timestamps[i:i+n]))
    
    return result

def tuple_list_to_str(data : list) -> list:
    return ["-".join(values) for values in data]

def transform_ngrams_list_to_seq_list(ngrams):
    """
    input :
    [['A','B'],['C','D']]
    
    output :
    ['A-B','C-D']
    """
    unique_rows = set(map(tuple, ngrams))
    return np.array(["-".join(row) for row in unique_rows])







# extract_correlation_data.py
import numpy as np

# Chargement de ton dictionnaire time (à adapter selon ton format de sauvegarde)
# Structure : time[gram][seq_without_last][last] = [(mean_context_gap, suffix_gap), ...]

def extract_correlation_pairs(time_dict: dict, mode: str = "mean") -> dict:
    """
    mode = "mean"     : corrèle mean_context_gap vs suffix_gap
    mode = "last"     : corrèle dernier_gap_contexte vs suffix_gap  
    mode = "both"     : retourne les deux en x (pour régression multiple)
    """
    assert mode in ("mean", "last"), f"mode inconnu : {mode}"
    result = {}
    for gram, contexts in time_dict.items():
        x_vals, y_vals = [], []
        for seq, suffixes in contexts.items():
            for last, pairs in suffixes.items():
                for triplet in pairs:
                    mean_ctx, suf_gap, last_gap = triplet
                    x = mean_ctx if mode == "mean" else last_gap
                    x_vals.append(float(x))
                    y_vals.append(float(suf_gap))
        result[int(gram)] = {"x": x_vals, "y": y_vals}
    return result

def compute_correlation_stats(x: list[float], y: list[float]) -> dict:
    """
    Calcule les statistiques de corrélation entre x et y.
    """
    from scipy.stats import spearmanr, kendalltau
    from scipy.stats import linregress
    
    
    x_arr = np.array(x, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[valid], y_arr[valid]

    
    
    # Filtrage des outliers extrêmes (au-delà du 99e percentile)
    x_p99 = np.percentile(x_arr, 99)
    y_p99 = np.percentile(y_arr, 99)
    mask  = (x_arr <= x_p99) & (y_arr <= y_p99)
    x_f, y_f = x_arr[mask], y_arr[mask]
    
    if len(x_f) < 10:
        return None
    
    # Garde-fou : si x ou y est constant, les corrélations ne sont pas définies
    if np.std(x_f) == 0 or np.std(y_f) == 0:
        return {
            "n_points"   : int(mask.sum()),
            "n_filtered" : int((~mask).sum()),
            "pearson_r"  : None,
            "r_squared"  : None,
            "spearman_r" : None,
            "spearman_p" : None,
            "kendall_tau": None,
            "kendall_p"  : None,
            "slope"      : None,
            "intercept"  : None,
            "x_median"   : round(float(np.median(x_f)), 1),
            "y_median"   : round(float(np.median(y_f)), 1),
            "sample_x"   : x_f[::max(1, len(x_f)//2000)].tolist(),
            "sample_y"   : y_f[::max(1, len(x_f)//2000)].tolist(),
            "error"      : "constant_input",
        }

    # Pearson
    pearson_r = float(np.corrcoef(x_f, y_f)[0, 1])

    # Spearman + Kendall
    spearman_r, spearman_p = spearmanr(x_f, y_f)
    kendall_t,  kendall_p  = kendalltau(x_f, y_f)

    # Régression via scipy.stats.linregress (robuste là où np.polyfit échoue)
    reg = linregress(x_f, y_f)
    y_pred    = reg.slope * x_f + reg.intercept
    ss_res    = float(np.sum((y_f - y_pred) ** 2))
    ss_tot    = float(np.sum((y_f - np.mean(y_f)) ** 2))
    r_squared = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "n_points"   : int(mask.sum()),
        "n_filtered" : int((~mask).sum()),
        "pearson_r"  : round(pearson_r, 4),
        "r_squared"  : round(r_squared, 4),
        "spearman_r" : round(float(spearman_r), 4),
        "spearman_p" : round(float(spearman_p), 6),
        "kendall_tau": round(float(kendall_t),  4),
        "kendall_p"  : round(float(kendall_p),  6),
        "slope"      : round(float(reg.slope),     4),
        "intercept"  : round(float(reg.intercept), 4),
        "x_median"   : round(float(np.median(x_f)), 1),
        "y_median"   : round(float(np.median(y_f)), 1),
        "sample_x"   : x_f[::max(1, len(x_f)//2000)].tolist(),
        "sample_y"   : y_f[::max(1, len(x_f)//2000)].tolist(),
    }

def generate_correlation_html(stats_by_gram: dict) -> str:
    import json as _json

    stats_json = _json.dumps(stats_by_gram)

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Corrélation gaps</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 24px; background: #f5f4f0; color: #1a1a1a; }}
  h1 {{ font-size: 20px; font-weight: 500; margin: 0 0 24px; }}
  .controls {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 20px; }}
  .controls label {{ font-size: 13px; color: #666; }}
  select, input[type=range] {{ font-size: 13px; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: #eeeee8; border-radius: 8px; padding: 14px 16px; }}
  .card-label {{ font-size: 12px; color: #666; margin: 0 0 4px; }}
  .card-value {{ font-size: 22px; font-weight: 500; margin: 0; }}
  .card-value.na {{ color: #aaa; font-size: 16px; }}
  .chart-wrap {{ background: #fff; border-radius: 10px; border: 0.5px solid #ddd; padding: 20px; margin-bottom: 24px; }}
  .chart-container {{ position: relative; width: 100%; height: 360px; }}
  .legend {{ display: flex; gap: 16px; font-size: 12px; color: #666; margin-bottom: 12px; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 4px; }}
  .legend-line {{ width: 18px; height: 2px; display: inline-block; vertical-align: middle; margin-right: 4px; }}
  .interp {{ font-size: 13px; color: #555; margin-top: 12px; min-height: 18px; }}
  .table-wrap {{ background: #fff; border-radius: 10px; border: 0.5px solid #ddd; padding: 20px; overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; font-weight: 500; padding: 6px 12px; border-bottom: 1px solid #eee; color: #444; }}
  td {{ padding: 6px 12px; border-bottom: 0.5px solid #f0f0f0; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }}
  .badge-strong {{ background: #c0dd97; color: #27500a; }}
  .badge-moderate {{ background: #b5d4f4; color: #0c447c; }}
  .badge-weak {{ background: #fac775; color: #633806; }}
  .badge-none {{ background: #eee; color: #888; }}
  .slider-val {{ font-size: 13px; font-weight: 500; min-width: 36px; display: inline-block; }}
</style>
</head>
<body>
<h1>Corrélation mean context gap → suffix gap</h1>

<div class="controls">
  <div>
    <label>Gram&nbsp;
      <select id="gram-sel"></select>
    </label>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <label>Cutoff outliers</label>
    <input type="range" id="cutoff" min="80" max="100" value="99" step="1">
    <span class="slider-val" id="cutoff-lbl">99%</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <label>Max points</label>
    <input type="range" id="maxpts" min="200" max="3000" value="1500" step="100">
    <span class="slider-val" id="maxpts-lbl">1500</span>
  </div>
</div>

<div class="metrics">
  <div class="card"><p class="card-label">Pearson r</p><p class="card-value" id="m-pearson">—</p></div>
  <div class="card"><p class="card-label">Spearman r</p><p class="card-value" id="m-spearman">—</p></div>
  <div class="card"><p class="card-label">Kendall τ</p><p class="card-value" id="m-kendall">—</p></div>
  <div class="card"><p class="card-label">R²</p><p class="card-value" id="m-r2">—</p></div>
  <div class="card"><p class="card-label">Pente</p><p class="card-value" id="m-slope">—</p></div>
  <div class="card"><p class="card-label">Intercept</p><p class="card-value" id="m-intercept">—</p></div>
  <div class="card"><p class="card-label">Points</p><p class="card-value" id="m-n">—</p></div>
  <div class="card"><p class="card-label">p-value (Sp.)</p><p class="card-value" id="m-pval">—</p></div>
</div>

<div class="chart-wrap">
  <div class="legend">
    <span><span class="legend-dot" style="background:#378ADD;opacity:0.6;"></span>données</span>
    <span><span class="legend-line" style="background:#E24B4A;"></span>régression</span>
  </div>
  <div class="chart-container">
    <canvas id="scatter" role="img" aria-label="Scatter plot corrélation gaps"></canvas>
  </div>
  <p class="interp" id="interp"></p>
</div>

<div class="table-wrap">
  <table id="summary-table">
    <thead>
      <tr>
        <th>Gram</th><th>Pearson r</th><th>Spearman r</th><th>Kendall τ</th>
        <th>R²</th><th>p-value</th><th>Points</th><th>Force</th>
      </tr>
    </thead>
    <tbody id="table-body"></tbody>
  </table>
</div>

<script>
const STATS = {stats_json};

function fmt(v, dec=3) {{
  if (v === null || v === undefined || isNaN(v)) return '—';
  return Number(v).toFixed(dec);
}}
function fmtTime(s) {{
  if (s === null || s === undefined) return '—';
  s = Math.round(s);
  if (s >= 3600) return (s/3600).toFixed(1) + 'h';
  if (s >= 60)   return Math.round(s/60) + 'm';
  return s + 's';
}}
function strength(r) {{
  if (r === null) return ['none','—'];
  const a = Math.abs(r);
  if (a >= 0.7) return ['strong','forte'];
  if (a >= 0.4) return ['moderate','modérée'];
  if (a >= 0.2) return ['weak','faible'];
  return ['none','très faible'];
}}
function interp(st) {{
  if (!st || st.error) return 'Input constant — corrélation non définie (gram=1 sans gaps de contexte).';
  const [,label] = strength(st.spearman_r);
  const dir = st.spearman_r >= 0 ? 'positive' : 'négative';
  const sig  = st.spearman_p < 0.05 ? ' (p < 0.05 — significatif)' : ' (p ≥ 0.05 — non significatif)';
  return `Spearman : corrélation ${{label}} ${{dir}}${{sig}}.`;
}}

function percentile(arr, p) {{
  const s = [...arr].sort((a,b)=>a-b);
  return s[Math.min(s.length-1, Math.floor(s.length*p/100))];
}}

const grams = Object.keys(STATS).map(Number).sort((a,b)=>a-b);
const sel = document.getElementById('gram-sel');
grams.forEach(g => {{
  const o = document.createElement('option');
  o.value = g; o.textContent = `Gram ${{g}}`;
  sel.appendChild(o);
}});

// Table de synthèse
const tbody = document.getElementById('table-body');
grams.forEach(g => {{
  const st = STATS[g];
  const [cls, label] = strength(st.spearman_r);
  const row = document.createElement('tr');
  row.innerHTML = `
    <td>${{g}}</td>
    <td>${{fmt(st.pearson_r)}}</td>
    <td>${{fmt(st.spearman_r)}}</td>
    <td>${{fmt(st.kendall_tau)}}</td>
    <td>${{fmt(st.r_squared)}}</td>
    <td>${{st.spearman_p < 0.001 ? '<0.001' : fmt(st.spearman_p, 4)}}</td>
    <td>${{(st.n_points||0).toLocaleString()}}</td>
    <td><span class="badge badge-${{cls}}">${{label}}</span></td>`;
  tbody.appendChild(row);
}});

let chart = null;

function render() {{
  const gram   = parseInt(sel.value);
  const cutoff = parseInt(document.getElementById('cutoff').value);
  const maxPts = parseInt(document.getElementById('maxpts').value);
  const st     = STATS[gram];

  // Métriques
  const setM = (id, val, dec=3) => {{
    const el = document.getElementById(id);
    if (val === null || val === undefined) {{ el.textContent='—'; el.className='card-value na'; }}
    else {{ el.textContent = typeof val === 'number' ? val.toFixed(dec) : val; el.className='card-value'; }}
  }};
  setM('m-pearson',   st.pearson_r);
  setM('m-spearman',  st.spearman_r);
  setM('m-kendall',   st.kendall_tau);
  setM('m-r2',        st.r_squared);
  setM('m-slope',     st.slope);
  setM('m-intercept', st.intercept, 0);
  setM('m-n',         st.n_points, 0);
  document.getElementById('m-n').textContent = (st.n_points||0).toLocaleString();
  document.getElementById('m-pval').textContent = 
    st.spearman_p < 0.001 ? '<0.001' : fmt(st.spearman_p, 4);
  document.getElementById('interp').textContent = interp(st);

  // Points
  let xs = st.sample_x || [], ys = st.sample_y || [];
  const xp = percentile(xs, cutoff), yp = percentile(ys, cutoff);
  let pts = xs.map(function(x,i){{ return {{x:x, y:ys[i]}}; }}).filter(function(p){{ return p.x<=xp && p.y<=yp; }});
  const step = Math.max(1, Math.floor(pts.length/maxPts));
  pts = pts.filter(function(_,i){{ return i%step===0; }});

  // Ligne de régression
  const regLine = (st.slope !== null && pts.length > 0) ? (() => {{
    const minX = Math.min(...pts.map(p=>p.x)), maxX = Math.max(...pts.map(p=>p.x));
    return [{{x:minX, y:st.slope*minX+st.intercept}}, {{x:maxX, y:st.slope*maxX+st.intercept}}];
  }})() : [];

  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('scatter'), {{
    data: {{
      datasets: [
        {{
          type: 'scatter', label: 'données', data: pts,
          backgroundColor: 'rgba(55,138,221,0.3)', pointRadius: 3, pointHoverRadius: 5,
        }},
        {{
          type: 'line', label: 'régression', data: regLine,
          borderColor: '#E24B4A', borderWidth: 1.5, pointRadius: 0, tension: 0,
        }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false, animation: {{ duration: 150 }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: ctx => `ctx moy: ${{fmtTime(ctx.parsed.x)}}  →  next gap: ${{fmtTime(ctx.parsed.y)}}`
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display:true, text:'mean context gap', font:{{size:12}} }},
               ticks: {{ callback: v => fmtTime(v) }} }},
        y: {{ title: {{ display:true, text:'suffix gap (Δt suivant)', font:{{size:12}} }},
               ticks: {{ callback: v => fmtTime(v) }} }}
      }}
    }}
  }});
}}

sel.addEventListener('change', render);
document.getElementById('cutoff').addEventListener('input', function() {{
  document.getElementById('cutoff-lbl').textContent = this.value + '%';
  render();
}});
document.getElementById('maxpts').addEventListener('input', function() {{
  document.getElementById('maxpts-lbl').textContent = this.value;
  render();
}});

render();
</script>
</body>
</html>"""

def get_day(filepath : Path) -> str:
    return filepath.name.split("_")[0]
def is_weekend(day : str) -> bool:
    """
    Determine if day is part of a weekend or not.
    
    ONLY FORMAT ACCEPTED FOR INPUT IS : "YYYY-MM-DD"
    """
    split_day = day.split("-")
    day_object = datetime.date(year = int(split_day[0]),
                  month = int(split_day[1]),
                  day = int(split_day[2]))
    return day_object.weekday() > 4
  
def save_csv(path : Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(file = path, mode="w", newline='') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(data)