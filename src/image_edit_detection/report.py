"""Build a concise result-oriented Markdown report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_course_report(
    template_path: Path,
    metrics_csv: Path,
    output_path: Path,
    figures_dir: Path,
    features_csv: Path | None = None,
) -> None:
    """Build a compact report that starts with the answer, not theory."""

    _ = template_path  # Kept for CLI compatibility.
    metrics = _read_csv(metrics_csv)
    features = _read_csv(features_csv) if features_csv is not None else pd.DataFrame()

    report = "\n\n".join(
        [
            "# Результат проверки изображений",
            _answer_section(metrics),
            _metrics_section(metrics),
            _dataset_section(features),
            _artifacts_section(figures_dir),
            _limitations_section(features),
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")


def _answer_section(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return (
            "## Краткий ответ\n\n"
            "Проверка еще не выполнена: файл `reports/metrics.csv` не найден или пустой."
        )

    ok_metrics = metrics[metrics.get("status", "") == "ok"].copy()
    if ok_metrics.empty:
        return (
            "## Краткий ответ\n\n"
            "Модель не была обучена: в метриках нет успешных экспериментов."
        )

    best = _best_row(ok_metrics)
    return (
        "## Краткий ответ\n\n"
        f"Лучший запуск: **{best['scope']} / {best['feature_group']}**. "
        f"На тестовой части accuracy = **{_fmt(best.get('accuracy'))}**, "
        f"F1 = **{_fmt(best.get('f1'))}**, ROC-AUC = **{_fmt(best.get('roc_auc'))}**.\n\n"
        "Это значит, что на текущем наборе данных Random Forest смог разделить изображения "
        "на `original` и `edited`. Если в запуске использовался маленький sample, результат "
        "нужно воспринимать как проверку пайплайна, а не как финальное качество модели."
    )


def _metrics_section(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "## Метрики\n\n_Метрики отсутствуют._"

    columns = ["scope", "feature_group", "rows", "test_rows", "accuracy", "precision", "recall", "f1", "roc_auc"]
    frame = metrics[[column for column in columns if column in metrics.columns]].copy()
    for column in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if column in frame.columns:
            frame[column] = frame[column].apply(_fmt)

    return "## Метрики\n\n" + _markdown_table(frame)


def _dataset_section(features: pd.DataFrame) -> str:
    if features.empty:
        return "## Данные\n\n_Файл `features/features.csv` не найден или пустой._"

    lines = ["## Данные", ""]
    counts = (
        features.groupby(["source", "label_name"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["source", "label_name"])
    )
    lines.append(_markdown_table(counts))

    if "manipulation_type" in features.columns:
        manipulations = (
            features.groupby(["source", "label_name", "manipulation_type"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["source", "label_name", "manipulation_type"])
        )
        lines.extend(["", "Типы манипуляций:", "", _markdown_table(manipulations)])

    return "\n".join(lines)


def _artifacts_section(figures_dir: Path) -> str:
    parts = ["## Файлы результата", ""]
    expected = [
        "metrics.csv",
        "public_metadata_classification_report.txt",
        "public_noise_classification_report.txt",
        "public_combined_classification_report.txt",
    ]
    parts.append("Основные файлы лежат в `reports/`:")
    parts.extend(f"- `{name}`" for name in expected)

    if figures_dir.exists():
        figure_count = len([path for path in figures_dir.rglob("*") if path.is_file() and path.name != ".gitkeep"])
        parts.append(f"- `reports/figures/` - графики и ELA-примеры, файлов: {figure_count}")

    parts.append("- `models/` - сохраненные `.joblib` модели Random Forest")
    parts.append("- `reports/predictions/` - отдельные отчеты по проверке одного изображения")
    return "\n".join(parts)


def _limitations_section(features: pd.DataFrame) -> str:
    rows = len(features) if not features.empty else 0
    warning = ""
    if rows and rows < 100:
        warning = (
            "\n\nВажно: сейчас данных мало. Такой результат подходит для демонстрации, "
            "но для надежной оценки стоит запускать проект на полном Columbia dataset "
            "или на более крупном наборе собственных изображений."
        )

    return (
        "## Вывод\n\n"
        "Для этой задачи подходит Random Forest: модель обучается на признаках metadata, "
        "ELA и шумовых паттернов, после чего предсказывает `original` или `edited`."
        f"{warning}"
    )


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _best_row(metrics: pd.DataFrame) -> pd.Series:
    scored = metrics.copy()
    scored["f1"] = pd.to_numeric(scored.get("f1"), errors="coerce")
    scored["roc_auc"] = pd.to_numeric(scored.get("roc_auc"), errors="coerce")
    return scored.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Нет данных._"

    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(rows)


def _fmt(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
