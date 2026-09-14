import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import train_test_split

from src.config import load_config, PROJECT_ROOT
from src.data.load import load_raw
from src.data.preprocess import split_xy, encode_labels
from src.features.build import add_features


def evaluate(config: dict | None = None):
    if config is None:
        config = load_config()

    rs = config["project"]["random_state"]
    target = config["data"]["target"]
    drop_cols = config["data"]["drop_columns"]
    date_col = config["data"].get("date_column")

    df = add_features(load_raw(config), date_column=date_col)
    X, y_str = split_xy(df, target, drop_cols)
    y, le = encode_labels(y_str)

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=rs, stratify=y,
    )

    model_dir = PROJECT_ROOT / config["paths"]["model_dir"]
    pipe = joblib.load(model_dir / "model.joblib")

    y_pred = pipe.predict(X_test)
    y_proba = None
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)

    figures_dir = PROJECT_ROOT / config["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(le.classes_)
    is_binary = n_classes == 2

    # --- Classification report ---
    print("\n--- Classification report ---")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0,
    ))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    print(f"[evaluate] Saved confusion_matrix.png")

    # --- Binary-only plots ---
    if is_binary and y_proba is not None:
        pos_proba = y_proba[:, 1]

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, pos_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(figures_dir / "roc_curve.png", dpi=120)
        plt.close(fig)
        print(f"[evaluate] ROC AUC: {roc_auc:.4f}")
        print(f"[evaluate] Saved roc_curve.png")

        # Precision-Recall curve
        prec, rec, _ = precision_recall_curve(y_test, pos_proba)
        ap = average_precision_score(y_test, pos_proba)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(rec, prec, label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(figures_dir / "pr_curve.png", dpi=120)
        plt.close(fig)
        print(f"[evaluate] Average Precision: {ap:.4f}")
        print(f"[evaluate] Saved pr_curve.png")

    # --- Feature importance ---
    try:
        model = pipe.named_steps["model"]
        prep = pipe.named_steps["prep"]
        names = prep.get_feature_names_out()
        importances = model.feature_importances_
        idx = np.argsort(importances)[-25:]
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.barh(np.array(names)[idx], importances[idx])
        ax.set_title("Top 25 feature importances")
        fig.tight_layout()
        fig.savefig(figures_dir / "feature_importance.png", dpi=120)
        plt.close(fig)
        print(f"[evaluate] Saved feature_importance.png")
    except Exception as e:
        print(f"[evaluate] Skipped feature importance plot: {e}")

    print(f"\n[evaluate] All plots saved to {figures_dir}")


if __name__ == "__main__":
    evaluate()
