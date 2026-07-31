"""
explain_movement_classifier.py
───────────────────────────────
Analyse l'impact des features sur les décisions des modèles via SHAP.

Niveaux d'analyse :
  1. Globale     — quelles features comptent le plus sur l'ensemble du test
  2. Par classe  — features qui poussent vers "move" vs "stay"
  3. Interactions — paires de features qui interagissent (XGBoost only)
  4. Locale      — explication d'une prédiction individuelle
  5. Erreurs     — features dominantes sur les FP et FN (cas d'échec du modèle)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
import joblib

# ══════════════════════════════════════════════════════════════════════════════
# CHEMINS
# ══════════════════════════════════════════════════════════════════════════════
MAIN_DIR    = Path(__file__).parent.parent.parent
FEATURE_DIR = MAIN_DIR / "results/predictions/movement_prediction/features"
MODEL_DIR   = MAIN_DIR / "results/predictions/movement_prediction/models"
OUTPUT_DIR  = MAIN_DIR / "results/predictions/movement_prediction/shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_CSV = FEATURE_DIR / "test_random_features.csv"

FEATURE_COLS = [
    "current_cell_duration_h",
    "current_streak",
    "recent_variability",
    "daily_transition_rate",
    "time_since_last_trans_h",
    "inter_transition_rhythm_h",
    "concentration",
    "momentum",
    "is_home_now",
    "is_activity_now",
    "home_hour_match",
    "activity_hour_match",
    "n_distinct_today",
]

# Noms courts pour les plots
FEATURE_LABELS = {
    "current_cell_duration_h":   "Duration in cell (h)",
    "current_streak":            "Streak length",
    "recent_variability":        "Recent variability",
    "daily_transition_rate":     "Daily transition rate",
    "time_since_last_trans_h":   "Time since last move (h)",
    "inter_transition_rhythm_h": "Inter-transition rhythm (h)",
    "concentration":             "Concentration (entropy)",
    "momentum":                  "Momentum",
    "is_home_now":               "Currently @ home",
    "is_activity_now":           "Currently @ activity",
    "home_hour_match":           "Home hour match",
    "activity_hour_match":       "Activity hour match",
    "n_distinct_today":          "Distinct cells today",
}

# Subsample pour accélérer SHAP sur grands datasets (None = tout)
MAX_SAMPLES = 20_000


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_test(path: Path, max_samples: int | None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path, sep=";").dropna(subset=FEATURE_COLS + ["label"])
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.int8)
    return X, y, df[FEATURE_COLS]


def load_model(name: str):
    path = MODEL_DIR / f"{name.lower()}_movement.joblib"
    print(f"  Loading {path.name} …")
    return joblib.load(path)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def feature_labels_list() -> list[str]:
    return [FEATURE_LABELS[f] for f in FEATURE_COLS]


def save_fig(name: str):
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {name}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. ANALYSE GLOBALE
#    mean(|SHAP|) par feature → classement d'importance "pondéré par impact réel"
#    Contrairement à feature_importances_, SHAP tient compte du signe et de la
#    distribution des valeurs.
# ══════════════════════════════════════════════════════════════════════════════

def plot_global_importance(shap_values: np.ndarray, model_name: str):
    """
    Bar chart : mean absolute SHAP par feature, triée par importance.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)
    labels = feature_labels_list()

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        [labels[i] for i in idx], mean_abs[idx],
        color="#534AB7", alpha=0.85
    )
    # Annotation valeur
    for bar, val in zip(bars, mean_abs[idx]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.set(title=f"{model_name} — Global feature importance (mean |SHAP|)",
           xlabel="mean |SHAP value|")
    plt.tight_layout()
    save_fig(f"{model_name.lower()}_global_importance.png")

    # Export JSON
    importance_dict = {FEATURE_COLS[i]: float(mean_abs[i]) for i in range(len(FEATURE_COLS))}
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    with open(OUTPUT_DIR / f"{model_name.lower()}_global_importance.json", "w") as f:
        json.dump(importance_dict, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 2. BEE SWARM PLOT
#    Chaque point = un sample. Axe X = valeur SHAP. Couleur = valeur de la feature.
#    Montre à la fois l'importance ET la direction de l'effet.
#    Ex : "streak élevé → SHAP négatif → pousse vers stay"
# ══════════════════════════════════════════════════════════════════════════════

def plot_beeswarm(shap_values: np.ndarray, X_df: pd.DataFrame, model_name: str):
    X_renamed = X_df.copy()
    X_renamed.columns = feature_labels_list()

    shap_exp = shap.Explanation(
        values=shap_values,
        data=X_renamed.values,
        feature_names=feature_labels_list(),
    )
    plt.figure()
    shap.plots.beeswarm(shap_exp, max_display=13, show=False)
    plt.title(f"{model_name} — SHAP beeswarm (direction & magnitude)")
    plt.tight_layout()
    save_fig(f"{model_name.lower()}_beeswarm.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DEPENDENCE PLOTS
#    Pour les 3 features les plus importantes : SHAP en fonction de la valeur
#    de la feature, coloré par la feature qui interagit le plus avec elle.
#    Révèle les effets non-linéaires et les interactions.
# ══════════════════════════════════════════════════════════════════════════════

def plot_dependences(shap_values: np.ndarray, X_df: pd.DataFrame, model_name: str, top_n: int = 3):
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
    labels   = feature_labels_list()

    fig, axes = plt.subplots(1, top_n, figsize=(6 * top_n, 5))
    if top_n == 1:
        axes = [axes]
    fig.suptitle(f"{model_name} — SHAP dependence plots (top {top_n} features)",
                 fontsize=12, fontweight="bold")

    for ax, feat_idx in zip(axes, top_idx):
        feat_name = labels[feat_idx]
        shap_feat = shap_values[:, feat_idx]
        feat_vals = X_df.iloc[:, feat_idx].values

        # Interaction color : feature la plus corrélée aux résidus SHAP
        residuals = shap_feat - np.polyval(np.polyfit(feat_vals, shap_feat, 1), feat_vals)
        interact_idx = np.argmax([
            abs(np.corrcoef(residuals, X_df.iloc[:, j].values)[0, 1])
            for j in range(len(FEATURE_COLS)) if j != feat_idx
        ])
        # Décalage d'index si on a sauté feat_idx
        if interact_idx >= feat_idx:
            interact_idx += 1
        color_vals = X_df.iloc[:, interact_idx].values
        interact_name = labels[interact_idx]

        sc = ax.scatter(feat_vals, shap_feat, c=color_vals,
                        cmap="coolwarm", alpha=0.4, s=8, rasterized=True)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set(xlabel=feat_name, ylabel="SHAP value",
               title=f"{feat_name}\n(color = {interact_name})")
        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    save_fig(f"{model_name.lower()}_dependence.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ANALYSE PAR OUTCOME : FP et FN
#    Compare les profils SHAP moyens sur :
#      - Vrais Positifs  (TP) : a bougé, prédit mouvement ✓
#      - Faux Positifs   (FP) : n'a pas bougé, prédit mouvement ✗
#      - Faux Négatifs   (FN) : a bougé, prédit stay ✗
#      - Vrais Négatifs  (TN) : n'a pas bougé, prédit stay ✓
#    Révèle quelles features trompent le modèle.
# ══════════════════════════════════════════════════════════════════════════════

def plot_error_analysis(shap_values: np.ndarray, y_true: np.ndarray,
                        y_pred: np.ndarray, model_name: str):
    labels = feature_labels_list()

    masks = {
        "TP": (y_true == 1) & (y_pred == 1),
        "TN": (y_true == 0) & (y_pred == 0),
        "FP": (y_true == 0) & (y_pred == 1),
        "FN": (y_true == 1) & (y_pred == 0),
    }
    colors = {"TP": "#1D9E75", "TN": "#534AB7", "FP": "#E85D24", "FN": "#BA7517"}
    linestyles = {"TP": "-", "TN": "-", "FP": "--", "FN": "--"}

    mean_shaps = {}
    for outcome, mask in masks.items():
        if mask.sum() > 0:
            mean_shaps[outcome] = shap_values[mask].mean(axis=0)

    # Radar-style : mean SHAP absolu par outcome, sur les features les plus importantes
    mean_abs_global = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_global)[::-1][:8]
    top_labels = [labels[i] for i in top_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{model_name} — Mean SHAP by prediction outcome",
                 fontsize=12, fontweight="bold")

    # Axe gauche : valeurs SHAP moyennes (signées) → direction de l'effet
    for outcome, mean_s in mean_shaps.items():
        axes[0].plot(
            [mean_s[i] for i in top_idx],
            marker="o", lw=2, ls=linestyles[outcome],
            color=colors[outcome], label=f"{outcome} (n={masks[outcome].sum():,})"
        )
    axes[0].axhline(0, color="gray", lw=0.8, ls=":")
    axes[0].set_xticks(range(len(top_idx)))
    axes[0].set_xticklabels(top_labels, rotation=35, ha="right", fontsize=9)
    axes[0].set(ylabel="Mean SHAP value (signed)", title="Direction of effect per outcome")
    axes[0].legend(fontsize=9)

    # Axe droit : |SHAP| moyen → magnitude de l'impact
    x = np.arange(len(top_idx))
    width = 0.2
    for i, (outcome, mean_s) in enumerate(mean_shaps.items()):
        axes[1].bar(
            x + i * width,
            [abs(mean_s[j]) for j in top_idx],
            width=width, label=f"{outcome}",
            color=colors[outcome], alpha=0.85
        )
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(top_labels, rotation=35, ha="right", fontsize=9)
    axes[1].set(ylabel="Mean |SHAP value|", title="Magnitude of impact per outcome")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    save_fig(f"{model_name.lower()}_error_analysis.png")

    # Export JSON résumé
    summary = {
        outcome: {
            FEATURE_COLS[i]: round(float(mean_shaps[outcome][i]), 6)
            for i in range(len(FEATURE_COLS))
        }
        for outcome in mean_shaps
    }
    with open(OUTPUT_DIR / f"{model_name.lower()}_error_shap.json", "w") as f:
        json.dump(summary, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 5. WATERFALL LOCAL
#    Explication d'une prédiction individuelle.
#    Choisit automatiquement un exemple représentatif par outcome.
# ══════════════════════════════════════════════════════════════════════════════

def plot_local_explanations(explainer, X_df: pd.DataFrame,
                            y_true: np.ndarray, y_pred: np.ndarray,
                            y_prob: np.ndarray, model_name: str):
    X_renamed = X_df.copy()
    X_renamed.columns = feature_labels_list()

    outcomes = {
        "FP_high_conf": np.where((y_true == 0) & (y_pred == 1) & (y_prob > 0.75))[0],
        "FN_high_conf": np.where((y_true == 1) & (y_pred == 0) & (y_prob < 0.25))[0],
        "TP_typical":   np.where((y_true == 1) & (y_pred == 1))[0],
        "TN_typical":   np.where((y_true == 0) & (y_pred == 0))[0],
    }

    for label, indices in outcomes.items():
        if len(indices) == 0:
            continue
        # Choisit l'exemple médian (le plus représentatif)
        sample_idx = indices[len(indices) // 2]
        shap_exp = explainer(X_renamed.iloc[[sample_idx]])

        plt.figure()
        shap.plots.waterfall(shap_exp[0], show=False)
        plt.title(f"{model_name} — {label}  "
                  f"(true={y_true[sample_idx]}, pred={y_pred[sample_idx]}, "
                  f"p={y_prob[sample_idx]:.2f})")
        plt.tight_layout()
        save_fig(f"{model_name.lower()}_local_{label}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. RÉSUMÉ TEXTE — insights clés
# ══════════════════════════════════════════════════════════════════════════════

def print_insights(shap_values: np.ndarray, X_df: pd.DataFrame,
                   y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    labels = feature_labels_list()
    mean_abs = np.abs(shap_values).mean(axis=0)
    top3_idx = np.argsort(mean_abs)[::-1][:3]

    print(f"\n  ┌─ {model_name} — Key insights ─────────────────────────────")

    # Top features globales
    print(f"  │  Top 3 most impactful features :")
    for rank, i in enumerate(top3_idx, 1):
        direction = "→ pushes toward MOVE" if shap_values[:, i].mean() > 0 else "→ pushes toward STAY"
        print(f"  │    {rank}. {labels[i]:<32} mean|SHAP|={mean_abs[i]:.4f}  {direction}")

    # Feature la plus trompeuse sur les FP
    fp_mask = (y_true == 0) & (y_pred == 1)
    if fp_mask.sum() > 0:
        fp_shap = np.abs(shap_values[fp_mask]).mean(axis=0)
        fp_top  = np.argmax(fp_shap)
        print(f"  │  Main feature driving False Positives : {labels[fp_top]}")

    # Feature la plus trompeuse sur les FN
    fn_mask = (y_true == 1) & (y_pred == 0)
    if fn_mask.sum() > 0:
        fn_shap = np.abs(shap_values[fn_mask]).mean(axis=0)
        fn_top  = np.argmax(fn_shap)
        print(f"  │  Main feature driving False Negatives : {labels[fn_top]}")

    # Feature la plus corrélée avec un effet non-monotone
    corr_lin = [abs(np.corrcoef(X_df.iloc[:, i].values, shap_values[:, i])[0, 1])
                for i in range(len(FEATURE_COLS))]
    nonlin_idx = np.argmin(corr_lin)
    print(f"  │  Most non-linear feature : {labels[nonlin_idx]} "
          f"(Pearson r={corr_lin[nonlin_idx]:.3f} with its SHAP)")
    print(f"  └────────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def analyse_model(name: str, X: np.ndarray, y_true: np.ndarray, X_df: pd.DataFrame):
    model = load_model(name)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"\n► Computing SHAP values for {name} …")
    if name == "XGBoost":
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # RF retourne [shap_class0, shap_class1] → on prend class 1 (move)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    print(f"  SHAP matrix : {shap_values.shape}")

    print(f"► Plotting — {name}")
    plot_global_importance(shap_values, name)
    plot_beeswarm(shap_values, X_df, name)
    plot_dependences(shap_values, X_df, name, top_n=3)
    plot_error_analysis(shap_values, y_true, y_pred, name)
    plot_local_explanations(explainer, X_df, y_true, y_pred, y_prob, name)
    print_insights(shap_values, X_df, y_true, y_pred, name)


def main():
    print("\n══ Movement classifier — SHAP explainability ══\n")

    print("► Loading test data")
    X, y, X_df = load_test(TEST_CSV, MAX_SAMPLES)
    print(f"  {len(y):,} samples  |  move rate = {y.mean():.3f}")

    for model_name in ["XGBoost"]:
        analyse_model(model_name, X, y, X_df)

    print(f"\n══ All outputs saved in {OUTPUT_DIR} ══\n")


if __name__ == "__main__":
    main()