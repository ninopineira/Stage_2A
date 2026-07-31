"""
train_movement_classifier.py
────────────────────────────
Entraîne un XGBoost et un RandomForest sur les features extraites par
MovementPredictor, évalue sur le split de test et sauvegarde :
  - Les deux modèles sérialisés (.joblib)
  - Un rapport de métriques complet (JSON + console)
  - Les courbes ROC, PR et l'importance des features (PNG)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    brier_score_loss, log_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import xgboost as xgb

# ══════════════════════════════════════════════════════════════════════════════
# CHEMINS — à adapter
# ══════════════════════════════════════════════════════════════════════════════
MAIN_DIR    = Path(__file__).parent.parent.parent
FEATURE_DIR = MAIN_DIR / "results/predictions/movement_prediction/features"
OUTPUT_DIR  = MAIN_DIR / "results/predictions/movement_prediction"
MODEL_DIR   = OUTPUT_DIR / "models_V6"
PLOT_DIR    = OUTPUT_DIR / "plots_V6"

TRAIN_CSV = FEATURE_DIR / "class1_train_random_features_V6.csv"
TEST_CSV  = FEATURE_DIR / "class1_test_random_features_V6.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    
    "mean_time_between_record",
    "record_per_hour",
    "number_of_time_in_first_cell",
    "moving",
    
    # -- Position vs ancre --
    "is_at_anchor_cell",
    "trip_phase",                  
    # -- Déclenchement du mouvement --
    "has_departed_today",   
    "first_departure_elapsed_h", 
    "time_in_anchor_before_dep_h",
    # -- Retour à l'ancre --
    "returned_to_anchor",   
    "time_since_return_h",
    # -- Dynamique du mouvement --
    "local_acceleration",         
    "n_complete_trips", 
    "current_trip_duration_h",
    # -- Profil de mobilité --
    "out_of_anchor_entropy"
]
LABEL_COL = "label"

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    print(f"  Loading {path.name} …", end=" ", flush=True)
    df = pd.read_csv(path, sep=";")
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int8)
    print(f"{len(y):,} samples  |  class balance : {y.mean():.3f} (move rate)")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# DÉFINITION DES MODÈLES
# ══════════════════════════════════════════════════════════════════════════════

def build_models(scale_pos_weight: float, max_depth : int) -> dict:
    """
    scale_pos_weight = n_negative / n_positive — compense le déséquilibre
    de classes pour XGBoost. RandomForest utilise class_weight='balanced'.
    """
    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500,
            max_depth=max_depth,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            early_stopping_rounds=30,
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(name: str, y_true: np.ndarray,
                    y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "model":            name,
        "accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "f1":               round(f1_score(y_true, y_pred), 4),
        "f1_macro":         round(f1_score(y_true, y_pred, average="macro"), 4),
        "precision":        round(precision_score(y_true, y_pred), 4),
        "recall":           round(recall_score(y_true, y_pred), 4),
        "roc_auc":          round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision":    round(average_precision_score(y_true, y_prob), 4),
        "log_loss":         round(log_loss(y_true, y_prob), 4),
        "brier_score":      round(brier_score_loss(y_true, y_prob), 4),
        # confusion matrix
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        # dérivées utiles pour le pipeline amont
        "specificity":      round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
        "miss_rate":        round(fn / (fn + tp) if (fn + tp) > 0 else 0, 4),
        "false_alarm_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {"XGBoost": "#E85D24", "RandomForest": "#1D9E75"}

def plot_roc_pr(results: dict, y_true: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ROC & Precision-Recall curves", fontsize=13, fontweight="bold")

    for name, res in results.items():
        color = COLORS[name]
        fpr, tpr, _ = roc_curve(y_true, res["y_prob"])
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{name}  (AUC = {res['metrics']['roc_auc']:.3f})")

        prec, rec, _ = precision_recall_curve(y_true, res["y_prob"])
        axes[1].plot(rec, prec, color=color, lw=2,
                     label=f"{name}  (AP = {res['metrics']['avg_precision']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title="ROC curve", xlim=[0, 1], ylim=[0, 1])
    axes[0].legend(loc="lower right")

    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision-Recall curve", xlim=[0, 1], ylim=[0, 1])
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → roc_pr.png")

def plot_feature_importance(results: dict, feature_names: list):
    fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle("Feature importance", fontsize=13, fontweight="bold")

    for ax, (name, res) in zip([axes], results.items()):
        importances = res["feature_importances"]
        idx = np.argsort(importances)
        color = COLORS[name]
        ax.barh(
            [feature_names[i] for i in idx],
            importances[idx],
            color=color, alpha=0.85
        )
        ax.set_title(name)
        ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → feature_importance.png")

def plot_confusion_matrices(results: dict, y_true: np.ndarray):
    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    fig.suptitle("Confusion matrices (test set)", fontsize=13, fontweight="bold")

    for ax, (name, res) in zip([axes], results.items()):
        cm = confusion_matrix(y_true, res["y_pred"])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Stay (0)", "Move (1)"])
        ax.set_yticklabels(["Stay (0)", "Move (1)"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → confusion_matrices.png")

def plot_calibration(results: dict, y_true: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    for name, res in results.items():
        frac_pos, mean_pred = calibration_curve(y_true, res["y_prob"], n_bins=20)
        ax.plot(mean_pred, frac_pos, marker="o", lw=2,
                color=COLORS[name], label=name)

    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
           title="Calibration curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → calibration.png")

def plot_threshold_analysis(results: dict, y_true: np.ndarray):
    """
    Pour chaque seuil θ, trace F1 / Recall / Precision / Specificity.
    Utile pour choisir θ optimal selon le coût des erreurs dans le pipeline.
    """
    thresholds = np.linspace(0.1, 0.9, 80)
    fig, axes = plt.subplots(1, 1, figsize=(14, 5), sharey=True)
    fig.suptitle("Metrics vs decision threshold", fontsize=13, fontweight="bold")

    for ax, (name, res) in zip([axes], results.items()):
        probs = res["y_prob"]
        f1s, precs, recs, specs = [], [], [], []
        for t in thresholds:
            pred = (probs >= t).astype(int)
            f1s.append(f1_score(y_true, pred, zero_division=0))
            precs.append(precision_score(y_true, pred, zero_division=0))
            recs.append(recall_score(y_true, pred, zero_division=0))
            cm = confusion_matrix(y_true, pred)
            tn, fp = cm[0, 0], cm[0, 1]
            specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        ax.plot(thresholds, f1s,   label="F1",          lw=2, color="#534AB7")
        ax.plot(thresholds, precs,  label="Precision",   lw=2, color="#E85D24")
        ax.plot(thresholds, recs,   label="Recall",      lw=2, color="#1D9E75")
        ax.plot(thresholds, specs,  label="Specificity", lw=2, color="#BA7517", ls="--")

        best_t = thresholds[np.argmax(f1s)]
        ax.axvline(best_t, color="gray", ls=":", lw=1.5,
                   label=f"Best F1 θ = {best_t:.2f}")
        ax.set(title=name, xlabel="Threshold θ", ylim=[0, 1])
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → threshold_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n══ Movement classifier — train & evaluate ══\n")

    # ── Chargement ────────────────────────────────────────────────────────────
    print("► Loading data")
    X_train, y_train = load_split(TRAIN_CSV)
    X_test,  y_test  = load_split(TEST_CSV)

    n_neg  = (y_train == 0).sum()
    n_pos  = (y_train == 1).sum()
    spw    = n_neg / n_pos  # scale_pos_weight pour XGBoost
    print(f"  Train : {n_neg:,} stay  |  {n_pos:,} move  |  spw = {spw:.2f}")



    for spw in [spw]: # [1.0,1.5,2.0,2.5,3.0,3.5]
        print(f"MODEL WITH SPW {spw}")
        models = build_models(scale_pos_weight=spw, max_depth=10)

        # ── Entraînement ──────────────────────────────────────────────────────────
        results = {}
        for name, model in models.items():
            print(f"\n► Training {name} …")
            t0 = time.time()

            if name == "XGBoost":
                # 10% du train comme validation set pour l'early stopping
                val_size = max(1, int(0.1 * len(X_train)))
                X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
                y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=50,
                )
            else:
                model.fit(X_train, y_train)

            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

            # ── Évaluation ────────────────────────────────────────────────────────
            y_prob = model.predict_proba(X_test)[:, 1]
            THRESHOLD = 0.65
            y_pred = (y_prob >= THRESHOLD).astype(int)

            metrics = compute_metrics(name, y_test, y_pred, y_prob)
            results[name] = {
                "model":              model,
                "y_pred":             y_pred,
                "y_prob":             y_prob,
                "metrics":            metrics,
                "feature_importances": (
                    model.feature_importances_
                    if hasattr(model, "feature_importances_")
                    else np.zeros(len(FEATURE_COLS))
                ),
            }

            # Résumé console
            print(f"\n  ┌─ {name} — test metrics ─────────────────────────")
            for k, v in metrics.items():
                if k not in ("model", "TP", "TN", "FP", "FN"):
                    print(f"  │  {k:<26} {v}")
            print(f"  │  Confusion : TP={metrics['TP']:,}  TN={metrics['TN']:,}  "
                f"FP={metrics['FP']:,}  FN={metrics['FN']:,}")
            print(f"  └────────────────────────────────────────────────")

            print(f"\n  Classification report :\n")
            print(classification_report(y_test, y_pred,
                                        target_names=["Stay (0)", "Move (1)"]))

            # Sauvegarde modèle
            model_path = MODEL_DIR / f"{name.lower()}_movement_extralate.joblib"
            joblib.dump(model, model_path)
            print(f"  Model saved → {model_path.name}")

        # ── Plots ─────────────────────────────────────────────────────────────────
        print("\n► Generating plots")
        plot_roc_pr(results, y_test)
        plot_feature_importance(results, FEATURE_COLS)
        plot_confusion_matrices(results, y_test)
        plot_calibration(results, y_test)
        plot_threshold_analysis(results, y_test)

        # ── Export JSON ───────────────────────────────────────────────────────────
        summary = {name: res["metrics"] for name, res in results.items()}
        json_path = OUTPUT_DIR / "movement_classifier_metrics.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n► Metrics saved → {json_path.name}")

        print("\n══ Done ══\n")


if __name__ == "__main__":
    main()


