# Результат проверки изображений

## Краткий ответ

{{BEST_MODEL_SUMMARY}}

## Данные

{{DATASET_SUMMARY}}

Основной источник обучения: `datasets/public`, который заполняется импортом внешних forensic-датасетов из `datasets/external`.

`datasets/generated` используется только как demo-режим для проверки пайплайна.

## Метрики обучения

{{METRICS_TABLE}}

## Артефакты

- `reports/metrics.csv` - таблица метрик по экспериментам.
- `reports/figures/` - confusion matrix, ROC curve, feature importance и ELA samples.
- `models/public_combined_random_forest.joblib` - основная модель для проверки одного изображения.
- `reports/predictions/latest_prediction.md` - отдельный отчет по последнему prediction.

## Что означает prediction

Модель Random Forest сначала обучается на изображениях с известной меткой `0=original`, `1=edited`.
После обучения для нового изображения извлекаются те же `meta_*` и `noise_*` признаки, затем модель выдает класс `original` или `edited` и вероятности классов.

Надежность prediction зависит от качества и размера датасета. Для финальной курсовой лучше обучать модель на полном Columbia dataset или на нескольких внешних forensic-датасетах.
