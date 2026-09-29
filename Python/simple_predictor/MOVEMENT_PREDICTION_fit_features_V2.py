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
FEATURE_DIR = MAIN_DIR / "results/predictions/movement_prediction_V2/features"
OUTPUT_DIR  = MAIN_DIR / "results/predictions/movement_prediction_V2"
MODEL_DIR   = OUTPUT_DIR / "models_V1"
PLOT_DIR    = OUTPUT_DIR / "plots_V1"

TRAIN_CSV = FEATURE_DIR / "class1_train_random_features.csv"
TEST_CSV  = FEATURE_DIR / "class1_test_random_features.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLS = [
    # -- Signaux de phase --
            "trip_phase",
            "is_at_anchor_cell",
            "returned_to_anchor",
            "has_departed_today",
            # -- Signaux locaux de rupture --
            "moving",
            "is_oscillating",
            "phase_transition_signal",
            "consecutive_non_anchor_records",
            "consecutive_anchor_records",
            "local_acceleration",
            # -- Densité et diversité locale --
            "recent_record_density",
            "last_n_distinct_cells",
            "anchor_dominance_ratio",
            # -- Profil journalier --
            "n_complete_trips",
            "out_of_anchor_entropy"
]
# Causal entropy features added to test the tutor's request ("integrate entropy").
ENTROPY_COLS = ["running_entropy", "running_cond_entropy", "running_move_rate", "running_pmax"]
FEATURE_COLS = BASE_COLS + ENTROPY_COLS
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
    # Single panel: only XGBoost is trained (the empty 2nd panel was a leftover from
    # the old XGBoost+RandomForest version). For the before/after comparison, see
    # plot_feature_importance_before_after().
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    fig.suptitle("Feature importance", fontsize=13, fontweight="bold")

    for name, res in results.items():
        importances = res["feature_importances"]
        idx = np.argsort(importances)
        ax.barh(
            [feature_names[i] for i in idx],
            importances[idx],
            color=COLORS[name], alpha=0.85
        )
        ax.set_title(name)
        ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → feature_importance.png")

def plot_confusion_matrices(results: dict, y_true: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Confusion matrices (test set)", fontsize=13, fontweight="bold")

    for ax, (name, res) in zip(axes, results.items()):
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle("Metrics vs decision threshold", fontsize=13, fontweight="bold")

    for ax, (name, res) in zip(axes, results.items()):
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

def compare_entropy_contribution():
    """Headline experiment: the SAME XGBoost, on the SAME data, trained WITHOUT vs
    WITH the causal entropy features. Prints the metrics and draws the before/after
    plots (overlaid ROC & PR curves + a grouped metric bar chart)."""
    tr = pd.read_csv(TRAIN_CSV, sep=";")
    te = pd.read_csv(TEST_CSV, sep=";")

    def run(cols):
        d_tr = tr.dropna(subset=cols + [LABEL_COL])
        d_te = te.dropna(subset=cols + [LABEL_COL])
        Xtr, ytr = d_tr[cols].values.astype(np.float32), d_tr[LABEL_COL].values.astype(np.int8)
        Xte, yte = d_te[cols].values.astype(np.float32), d_te[LABEL_COL].values.astype(np.int8)
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        model = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.03,
                                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                  scale_pos_weight=spw, eval_metric="logloss",
                                  random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        return {"y_true": yte, "y_prob": prob, "n_features": len(cols),
                "model": model, "cols": cols,
                "auc": roc_auc_score(yte, prob),
                "ap": average_precision_score(yte, prob),
                "f1": f1_score(yte, pred)}

    res = {"without entropy": run(BASE_COLS), "with entropy": run(FEATURE_COLS)}

    print("\n══ Does the causal entropy help the move/stay classifier? ══")
    for tag, r in res.items():
        print(f"  {tag:<16} AUC={r['auc']:.4f}  AP={r['ap']:.4f}  F1={r['f1']:.4f}   ({r['n_features']} features)")
    a, b = res["without entropy"], res["with entropy"]
    print(f"  {'gain (delta)':<16} AUC={b['auc']-a['auc']:+.4f}  AP={b['ap']-a['ap']:+.4f}  F1={b['f1']-a['f1']:+.4f}\n")

    plot_before_after(res)
    plot_feature_importance_before_after(res)
    plot_threshold_analysis_before_after(res)


def plot_feature_importance_before_after(res):
    """Feature importance without (left) vs with (right) the entropy features.
    The entropy features are highlighted on the right panel."""
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, tag in zip(axes, ["without entropy", "with entropy"]):
        r = res[tag]
        cols, imp = r["cols"], r["model"].feature_importances_
        order = np.argsort(imp)                     # ascending -> largest on top of barh
        names = [cols[i] for i in order]
        colors = ["#2a8a62" if nm in ENTROPY_COLS else "#6b7d8a" for nm in names]
        ax.barh(names, imp[order], color=colors)
        ax.set_title(f"{tag}  ({len(cols)} features)", fontweight="bold")
        ax.set_xlabel("Importance")
        ax.tick_params(axis="y", labelsize=8)
    axes[1].legend(handles=[Patch(color="#2a8a62", label="entropy features"),
                            Patch(color="#6b7d8a", label="base features")],
                   loc="lower right", fontsize=9)
    fig.suptitle("Feature importance — before (without entropy) vs after (with entropy)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = PLOT_DIR / "movement_entropy_feature_importance_before_after.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.show()


def plot_threshold_analysis_before_after(res):
    """The predecessor's threshold analysis (F1 / Precision / Recall / Specificity vs
    the decision threshold), split into two panels: left = before (without entropy),
    right = after (with entropy). The dotted grey line marks the best-F1 threshold."""
    thresholds = np.linspace(0.1, 0.9, 80)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, tag in zip(axes, ["without entropy", "with entropy"]):
        r = res[tag]
        yt, yp = r["y_true"], r["y_prob"]
        f1s, precs, recs, specs = [], [], [], []
        for t in thresholds:
            pred = (yp >= t).astype(int)
            f1s.append(f1_score(yt, pred, zero_division=0))
            precs.append(precision_score(yt, pred, zero_division=0))
            recs.append(recall_score(yt, pred, zero_division=0))
            tn = int(((yt == 0) & (pred == 0)).sum())
            fp = int(((yt == 0) & (pred == 1)).sum())
            specs.append(tn / (tn + fp) if (tn + fp) else 0.0)
        ax.plot(thresholds, f1s,   label="F1",          lw=2, color="#534AB7")
        ax.plot(thresholds, precs, label="Precision",   lw=2, color="#E85D24")
        ax.plot(thresholds, recs,  label="Recall",      lw=2, color="#1D9E75")
        ax.plot(thresholds, specs, label="Specificity", lw=2, color="#BA7517", ls="--")
        best_t = thresholds[int(np.argmax(f1s))]
        ax.axvline(best_t, color="grey", ls=":", lw=1.5, label=f"best-F1 θ = {best_t:.2f}")
        ax.set(title=f"{tag}  ({r['n_features']} features)", xlabel="Decision threshold θ",
               xlim=(0.1, 0.9), ylim=(0, 1))
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(fontsize=8, loc="center right")
    axes[0].set_ylabel("Score")
    fig.suptitle("Metrics vs decision threshold — before vs after entropy",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = PLOT_DIR / "movement_entropy_threshold_before_after.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.show()


def plot_before_after(res):
    """Overlaid ROC & PR curves + grouped metric bars, without vs with entropy."""
    col = {"without entropy": "#6b7d8a", "with entropy": "#2a8a62"}   # grey vs green
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for tag, r in res.items():
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        axes[0].plot(fpr, tpr, color=col[tag], lw=2, label=f"{tag} (AUC={r['auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC",
                xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(loc="lower right", fontsize=9)

    for tag, r in res.items():
        prec, rec, _ = precision_recall_curve(r["y_true"], r["y_prob"])
        axes[1].plot(rec, prec, color=col[tag], lw=2, label=f"{tag} (AP={r['ap']:.3f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall",
                xlim=(0, 1), ylim=(0, 1))
    axes[1].legend(loc="lower left", fontsize=9)

    metrics, labels = ["auc", "ap", "f1"], ["ROC-AUC", "AP", "F1"]
    x, width = np.arange(len(metrics)), 0.38
    for i, (tag, r) in enumerate(res.items()):
        bars = axes[2].bar(x + (i - 0.5) * width, [r[m] for m in metrics], width,
                           label=tag, color=col[tag])
        axes[2].bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Metrics (test)")
    axes[2].legend(fontsize=9)

    fig.suptitle("Move/stay classifier — before (base features) vs after (+ entropy)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = PLOT_DIR / "movement_entropy_before_after.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.show()


def main():
    compare_entropy_contribution()
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
        models = build_models(scale_pos_weight=spw, max_depth=6)

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


