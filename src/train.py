from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.preprocess import FEATURE_COLUMNS, prepare_data


def build_model_pipeline(preprocessor) -> Pipeline:
    """Create the end-to-end preprocessing + model pipeline."""
    classifier = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def get_param_grid() -> Dict[str, list]:
    """Small hyperparameter grid to stay lightweight and fast."""
    return {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 8, 12],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }


def evaluate_model(y_true, y_pred, class_names) -> Tuple[float, np.ndarray, Dict]:
    """Compute core evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return accuracy, cm, report


def save_artifacts(model, label_encoder, metrics: Dict, output_dir: Path) -> None:
    """Persist model, encoder, and metrics to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pkl"
    encoder_path = output_dir / "label_encoder.pkl"
    metrics_path = output_dir / "metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def train(data_path: Path, output_dir: Path, experiment_name: str) -> None:
    """Train, evaluate, log with MLflow, and save artifacts."""
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = prepare_data(data_path)

    pipeline = build_model_pipeline(preprocessor)
    param_grid = get_param_grid()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="random_forest_training"):
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        class_names = list(label_encoder.classes_)
        accuracy, cm, report = evaluate_model(y_test, y_pred, class_names)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("cv_best_score", float(grid_search.best_score_))

        # Log macro averages for quick model quality tracking.
        macro_avg = report.get("macro avg", {})
        if macro_avg:
            mlflow.log_metric("precision_macro", float(macro_avg.get("precision", 0.0)))
            mlflow.log_metric("recall_macro", float(macro_avg.get("recall", 0.0)))
            mlflow.log_metric("f1_macro", float(macro_avg.get("f1-score", 0.0)))

        mlflow.sklearn.log_model(best_model, artifact_path="model")

        metrics = {
            "accuracy": float(accuracy),
            "cv_best_score": float(grid_search.best_score_),
            "best_params": grid_search.best_params_,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "feature_columns": FEATURE_COLUMNS,
            "classes": class_names,
        }

        save_artifacts(best_model, label_encoder, metrics, output_dir)

        print("Training complete.")
        print(f"Best Params: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mental Health Risk model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/mental_health.csv"),
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model"),
        help="Directory for saved model artifacts",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="mental-health-risk-prediction",
        help="MLflow experiment name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_path, args.output_dir, args.experiment_name)
