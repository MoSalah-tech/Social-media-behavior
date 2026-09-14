import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import load_config, PROJECT_ROOT
from src.data.load import load_raw
from src.data.validate import validate
from src.data.preprocess import (
    build_preprocessor, detect_feature_types, split_xy, encode_labels,
)
from src.features.build import add_features


def compute_defaults(df: pd.DataFrame) -> dict:
    """Compute a sensible default (median/mode) for every feature."""
    defaults = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = float(df[col].median())
        else:
            mode = df[col].mode()
            defaults[col] = str(mode.iloc[0]) if len(mode) else ""
    return defaults


def build_model_params(base_params: dict, n_classes: int) -> dict:
    """Auto-set objective / eval_metric / num_class based on task type."""
    params = dict(base_params)
    if n_classes == 2:
        params.pop("num_class", None)
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
        params["num_class"] = n_classes
    return params


def compute_metrics(y_true, y_pred, y_proba, n_classes: int) -> dict:
    """Metrics depend on binary vs multi-class."""
    metrics = {"test_accuracy": float(accuracy_score(y_true, y_pred))}
    if n_classes == 2:
        metrics["test_f1"] = float(f1_score(y_true, y_pred))
        metrics["test_f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        if y_proba is not None:
            metrics["test_roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
    else:
        metrics["test_f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        metrics["test_f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
    return metrics


def train(config: dict | None = None):
    if config is None:
        config = load_config()

    rs = config["project"]["random_state"]
    target = config["data"]["target"]
    drop_cols = config["data"]["drop_columns"]
    date_col = config["data"].get("date_column")

    # 1. Load + feature engineering + validate
    df = load_raw(config)
    df = add_features(df, date_column=date_col)
    validate(df, target)

    # 2. Split X / y
    X, y_str = split_xy(df, target, drop_cols)
    y, label_encoder = encode_labels(y_str)
    n_classes = len(label_encoder.classes_)
    task = "binary" if n_classes == 2 else f"multiclass ({n_classes})"
    print(f"[train] task: {task}")
    print(f"[train] classes: {list(label_encoder.classes_)}")
    print(f"[train] class balance: "
          f"{dict(y_str.value_counts(normalize=True).round(3))}")

    # 3. Train / val / test (stratified)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=rs, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=config["split"]["val_size"],
        random_state=rs, stratify=y_train_full,
    )

    # 4. Feature types
    numeric_cols, categorical_cols = detect_feature_types(df, target, drop_cols)
    print(f"[train] numeric={len(numeric_cols)}, categorical={len(categorical_cols)}")

    # 5. Build pipeline
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model_params = build_model_params(config["model"]["params"], n_classes)
    model = XGBClassifier(**model_params)

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    # 6. Defaults for inference (computed from full feature set)
    defaults = compute_defaults(X)

    # 7. MLflow
    mlflow_dir = PROJECT_ROOT / config["paths"]["mlflow_dir"]
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_dir.as_uri())
    mlflow.set_experiment(config["project"]["name"])

    with mlflow.start_run() as run:
        mlflow.log_params({
            k: v for k, v in model_params.items()
            if isinstance(v, (int, float, str, bool))
        })
        mlflow.log_param("n_numeric", len(numeric_cols))
        mlflow.log_param("n_categorical", len(categorical_cols))
        mlflow.log_param("n_classes", n_classes)
        mlflow.log_param("task", task)
        mlflow.log_param("target", target)

        # 8. Cross-validation
        cv = StratifiedKFold(
            n_splits=config["training"]["cv_folds"],
            shuffle=True, random_state=rs,
        )
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=cv,
            scoring=config["training"]["scoring"], n_jobs=-1,
        )
        mlflow.log_metric("cv_score_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_score_std", float(cv_scores.std()))
        print(f"[train] CV {config['training']['scoring']}: "
              f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # 9. Fit
        pipe.fit(X_train, y_train)

        # 10. Evaluate
        y_pred = pipe.predict(X_test)
        try:
            y_proba = pipe.predict_proba(X_test)
        except AttributeError:
            y_proba = None

        metrics = compute_metrics(y_test, y_pred, y_proba, n_classes)
        mlflow.log_metrics(metrics)
        print("[train] Test metrics:")
        for k, v in metrics.items():
            print(f"           {k:<20} {v:.4f}")

        # 11. Save artifacts
        model_dir = PROJECT_ROOT / config["paths"]["model_dir"]
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipe, model_dir / "model.joblib")
        joblib.dump(label_encoder, model_dir / "label_encoder.joblib")

        schema = {
            "target": target,
            "task": task,
            "classes": list(label_encoder.classes_),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "drop_columns": drop_cols,
            "date_column": date_col,
            "defaults": defaults,
            "metrics": metrics,
        }
        with open(model_dir / "schema.json", "w") as f:
            json.dump(schema, f, indent=2)

        report = classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        with open(model_dir / "classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

        mlflow.sklearn.log_model(pipe, "model")
        mlflow.log_artifact(str(model_dir / "schema.json"))

        print(f"[train] Saved model to {model_dir / 'model.joblib'}")
        print(f"[train] MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    train()
