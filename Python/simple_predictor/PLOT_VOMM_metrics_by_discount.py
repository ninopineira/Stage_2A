import json
import re
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
# 1. Nom de votre fichier JSON unique

MAIN_DIR = Path(__file__).parent.parent.parent
json_file_path = MAIN_DIR / "results/predictions/simple_predictor/metrics/V1/V1_discount_study.json"

# 2. Lecture et extraction des données
with open(json_file_path, 'r') as f:
    full_data = json.load(f)

data_list = []

for key, metrics in full_data.items():
    # Extraction de la valeur numérique après le dernier tiret du bas (ex: 0.1 depuis "rdm_nrepeat_0.1")
    match = re.search(r"(\d+\.\d+)$", key)
    if match:
        x_val = float(match.group(1))
        
        # Récupération des métriques associées à cette clé
        data_list.append({
            'X': x_val,
            'ACC@1': metrics.get('ACC@1'),
            'ACC@3': metrics.get('ACC@3'),
            'ACC@5': metrics.get('ACC@5'),
            'ACC@10': metrics.get('ACC@10'),
            'log-l_mean': metrics.get('log-l_mean')
        })

# 3. Création du DataFrame et tri automatique par les valeurs de X
df = pd.DataFrame(data_list)
df = df.sort_values(by='X').reset_index(drop=True)

# 4. Génération des graphiques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1 : plt.Axes

# --- Graphique 1 : Évolution de l'Accuracy (ACC@K) ---
ax1.plot(df['X'], df['ACC@1'], marker='o', label='ACC@1', linewidth=2)
ax1.plot(df['X'], df['ACC@3'], marker='s', label='ACC@3', linewidth=2)
ax1.plot(df['X'], df['ACC@5'], marker='^', label='ACC@5', linewidth=2)
ax1.plot(df['X'], df['ACC@10'], marker='x', label='ACC@10', linewidth=2)

ax1.set_title("Evolution of accuracy (ACC@K) as a function of the discount")
ax1.set_xlabel("Discount value")
ax1.set_ylabel("Score Accuracy")
ax1.set_xticks([k/10 for k in range(1,10)])
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# --- Graphique 2 : Évolution de la log-l_mean ---
ax2.plot(df['X'], df['log-l_mean'], marker='o', color='purple', linewidth=2, linestyle='--')

ax2.set_title("Evolution of Mean log-loss evolution as a function of the discount")
ax2.set_xlabel("Discount value")
ax2.set_ylabel("Mean log loss")
ax2.set_xticks([k/10 for k in range(1,10)])
ax2.grid(True, linestyle='--', alpha=0.6)

# Ajustement automatique des espaces et sauvegarde du graphique
plt.tight_layout()
plot_output = MAIN_DIR / "results/predictions/simple_predictor/plots/vomm_combined_metrics_plot.png"
plot_output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_output, dpi=300)
plt.show()