"""Prediction utilities for a single unknown image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from image_edit_detection.features import extract_image_features


@dataclass(frozen=True)
class PredictionResult:
    image_path: Path
    model_path: Path
    prediction: str
    probabilities: dict[str, float]
    scope: str
    feature_group: str


def predict_image(image_path: Path, model_path: Path) -> PredictionResult:
    """Load a trained model and predict whether one image is original or edited."""

    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    payload = joblib.load(model_path)
    model = payload["model"]
    feature_columns = list(payload["feature_columns"])
    label_mapping = payload.get("label_mapping", {"original": 0, "edited": 1})
    reverse_mapping = {int(value): str(key) for key, value in label_mapping.items()}

    features = extract_image_features(image_path)
    frame = pd.DataFrame([features])
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0
    frame = frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

    predicted_label = int(model.predict(frame)[0])
    prediction = reverse_mapping.get(predicted_label, str(predicted_label))

    probabilities: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)[0]
        classes = [int(value) for value in model.classes_]
        probabilities = {
            reverse_mapping.get(class_id, str(class_id)): float(probability)
            for class_id, probability in zip(classes, proba)
        }

    return PredictionResult(
        image_path=image_path,
        model_path=model_path,
        prediction=prediction,
        probabilities=probabilities,
        scope=str(payload.get("scope", "")),
        feature_group=str(payload.get("feature_group", "")),
    )


def save_prediction_report(
    result: PredictionResult,
    output_path: Path,
    visualization_path: Path | None = None,
) -> None:
    """Save a compact Markdown report for one prediction."""

    probability_lines = [
        f"- `{label}`: {probability:.3f}" for label, probability in result.probabilities.items()
    ]
    if not probability_lines:
        probability_lines = ["- probabilities unavailable"]

    visualization_lines: list[str] = []
    if visualization_path is not None:
        visualization_lines = [
            "ELA-псевдомаска:",
            "",
            f"![ELA evidence]({_markdown_path(output_path, visualization_path)})",
            "",
            "Красная область показывает повышенный ELA-сигнал. "
            "Это подсказка для анализа, а не точная ground-truth маска редактирования.",
            "",
        ]

    text = "\n".join(
        [
            "# Результат проверки изображения",
            "",
            f"Image: `{result.image_path}`",
            f"Model: `{result.model_path.name}`",
            f"Training scope: `{result.scope}`",
            f"Feature group: `{result.feature_group}`",
            "",
            f"Prediction: **{result.prediction}**",
            "",
            "Вероятности:",
            *probability_lines,
            "",
            *visualization_lines,
            "Важно: prediction надежен настолько, насколько надежен обучающий датасет. "
            "Для финального результата используй полный forensic dataset.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _markdown_path(report_path: Path, artifact_path: Path) -> str:
    try:
        return artifact_path.resolve().relative_to(report_path.parent.resolve()).as_posix()
    except ValueError:
        return artifact_path.as_posix()
