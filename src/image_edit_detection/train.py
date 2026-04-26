"""Random Forest training and evaluation utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from image_edit_detection.features import feature_columns


DEFAULT_SCOPES = ("generated", "public", "all")
DEFAULT_FEATURE_GROUPS = ("metadata", "noise", "combined")


def run_experiments(
    feature_csv: Path,
    reports_dir: Path,
    models_dir: Path,
    scopes: Iterable[str] = DEFAULT_SCOPES,
    feature_groups: Iterable[str] = DEFAULT_FEATURE_GROUPS,
    random_state: int = 42,
    test_size: float = 0.25,
    tune: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run the required dataset-scope and feature-group experiments."""

    frame = pd.read_csv(feature_csv)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for scope in scopes:
        for feature_group in feature_groups:
            try:
                result = train_single_experiment(
                    frame=frame,
                    scope=scope,
                    feature_group=feature_group,
                    reports_dir=reports_dir,
                    figures_dir=figures_dir,
                    models_dir=models_dir,
                    random_state=random_state,
                    test_size=test_size,
                    tune=tune,
                    n_jobs=n_jobs,
                )
                rows.append(result)
            except ValueError as exc:
                rows.append(
                    {
                        "scope": scope,
                        "feature_group": feature_group,
                        "status": "skipped",
                        "error": str(exc),
                    }
                )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(reports_dir / "metrics.csv", index=False)
    return metrics


def train_single_experiment(
    frame: pd.DataFrame,
    scope: str,
    feature_group: str,
    reports_dir: Path,
    figures_dir: Path,
    models_dir: Path,
    random_state: int = 42,
    test_size: float = 0.25,
    tune: bool = True,
    n_jobs: int = 1,
) -> dict[str, object]:
    """Train and evaluate one Random Forest experiment."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    data = _filter_scope(frame, scope)
    if data.empty:
        raise ValueError(f"No rows for scope {scope!r}")
    if data["label"].nunique() < 2:
        raise ValueError(f"Scope {scope!r} does not contain both classes")

    columns = feature_columns(data, feature_group)
    y = data["label"].astype(int)
    if y.value_counts().min() < 2:
        raise ValueError("Need at least two images per class for stratified evaluation")

    X = data[columns].replace([np.inf, -np.inf], np.nan)
    test_count = _resolve_test_count(y, test_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_count,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )

    estimator, best_params = _fit_random_forest(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        tune=tune,
        n_jobs=n_jobs,
    )

    y_pred = estimator.predict(X_test)
    y_score = estimator.predict_proba(X_test)[:, 1]
    roc_auc = _safe_roc_auc(y_test, y_score)
    experiment_name = f"{scope}_{feature_group}"

    _save_confusion_matrix(y_test, y_pred, figures_dir / f"{experiment_name}_confusion_matrix.png")
    _save_roc_curve(y_test, y_score, figures_dir / f"{experiment_name}_roc_curve.png", roc_auc)
    _save_feature_importances(
        estimator=estimator,
        columns=columns,
        output_path=figures_dir / f"{experiment_name}_feature_importance.png",
    )
    _save_classification_report(
        y_test=y_test,
        y_pred=y_pred,
        output_path=reports_dir / f"{experiment_name}_classification_report.txt",
    )

    model_payload = {
        "model": estimator,
        "feature_columns": columns,
        "scope": scope,
        "feature_group": feature_group,
        "label_mapping": {"original": 0, "edited": 1},
    }
    joblib.dump(model_payload, models_dir / f"{experiment_name}_random_forest.joblib")

    return {
        "scope": scope,
        "feature_group": feature_group,
        "status": "ok",
        "rows": len(data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": len(columns),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "best_params": json.dumps(best_params, ensure_ascii=True),
    }


def _fit_random_forest(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool,
    n_jobs: int,
) -> tuple[Pipeline, dict[str, object]]:
    if not tune:
        pipeline.set_params(
            model__n_estimators=300,
            model__max_depth=None,
            model__min_samples_leaf=1,
            model__max_features="sqrt",
        )
        pipeline.fit(X_train, y_train)
        return pipeline, {
            "model__n_estimators": 300,
            "model__max_depth": None,
            "model__min_samples_leaf": 1,
            "model__max_features": "sqrt",
        }

    min_class_count = int(y_train.value_counts().min())
    cv_splits = min(5, min_class_count)
    if cv_splits < 2:
        raise ValueError("Not enough training samples per class for GridSearchCV")

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 12],
        "model__min_samples_leaf": [1, 3],
        "model__max_features": ["sqrt", "log2"],
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_splits,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def _filter_scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return frame.copy()
    if scope not in {"generated", "public"}:
        raise ValueError("scope must be generated, public, or all")
    return frame[frame["source"] == scope].copy()


def _resolve_test_count(y: pd.Series, test_size: float) -> int:
    class_count = int(y.nunique())
    total = len(y)
    requested = max(class_count, int(math.ceil(total * test_size)))
    max_allowed = total - class_count
    test_count = min(requested, max_allowed)
    if test_count < class_count:
        raise ValueError("Not enough rows to create a stratified train/test split")
    return test_count


def _safe_roc_auc(y_test: pd.Series, y_score: np.ndarray) -> float:
    if len(set(y_test)) < 2:
        return float("nan")
    return float(roc_auc_score(y_test, y_score))


def _save_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["original", "edited"],
        yticklabels=["original", "edited"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_roc_curve(
    y_test: pd.Series,
    y_score: np.ndarray,
    output_path: Path,
    roc_auc: float,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(set(y_test)) >= 2:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        label = f"ROC AUC = {roc_auc:.3f}" if math.isfinite(roc_auc) else "ROC"
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_feature_importances(
    estimator: Pipeline,
    columns: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    model = estimator.named_steps["model"]
    importances = pd.Series(model.feature_importances_, index=columns)
    top = importances.sort_values(ascending=False).head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(top))))
    top.plot(kind="barh", ax=ax, color="#3b82f6")
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_classification_report(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    text = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["original", "edited"],
        zero_division=0,
    )
    output_path.write_text(text, encoding="utf-8")
