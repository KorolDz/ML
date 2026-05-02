# Forensic-анализ редактирования изображений

Курсовой проект по машинному обучению в направлении **"Обнаружение фейков и цифровых манипуляций в медиафайлах"**.

Выбранная подтема: **анализ метаданных и шумовых паттернов изображений для определения следов редактирования в графических редакторах**.

Задача проекта - обучить модель бинарной классификации `original` / `edited` на forensic-признаках изображения: metadata, EXIF, Error Level Analysis, шумовой остаток и JPEG-артефакты. Основной метод - **Random Forests / Случайные леса**.

Проект не решает задачи GAN/diffusion detection, DeepFake-видео или NLP fake-news: здесь используется классическое признаковое описание изображения и табличная ML-модель.

## Структура

```text
datasets/
  external/             # сюда вручную кладутся распакованные внешние датасеты
  public/
    original/           # импортированные authentic/original изображения
    edited/             # импортированные spliced/forged/tampered изображения
    manifest.csv        # единый manifest для обучения
  generated/            # optional demo dataset, не основной источник обучения
  source/               # optional исходники для generated demo
features/               # generated features.csv
models/                 # trained *.joblib models
reports/
  figures/              # confusion matrix, ROC, feature importance, ELA samples
  predictions/          # отчеты по проверке одного изображения
src/image_edit_detection/
scripts/
tests/
```

В git должны попадать код, тесты, README, шаблоны и `.gitkeep`. Изображения, признаки, модели и отчеты после запуска игнорируются через `.gitignore`.

## Установка

```powershell
cd H:\ML\ML
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Download datasets

Скачай один или несколько датасетов и распакуй их в `datasets/external/<dataset_name>`.

Рекомендуемый первый датасет:

- **Columbia Uncompressed Image Splicing Detection Dataset**  
  Download form: https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/dlform.html  
  Ожидаемые папки после распаковки: `4cam_auth`, `4cam_splc`.

Дополнительные варианты:

- **Columbia Image Splicing Detection Evaluation Dataset**  
  Page: https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm  
  Download form: https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/dlform.html
- **CoMoFoD**  
  https://www.vcl.fer.hr/comofod/download.html
- **COVERAGE**  
  https://github.com/wenbihan/coverage
- **CASIA v2**  
  Kaggle mirror: https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset  
  Используй только если лицензия/источник подходят для твоей курсовой.
- **TIF-пары вида `1.tif` / `1t.tif`**  
  Импортируются через `--kind tif-pairs`: файл без `t` считается `original`, файл с суффиксом `t` считается `edited`.

Пример структуры:

```text
datasets/external/
  columbia/
    4cam_auth/
    4cam_splc/
  casia/
    Au/
    Tp/
  comofod/
  coverage/
  image/
```

## Основной workflow

### 1. Импортировать внешний датасет

Для Columbia:

```powershell
python scripts/02_import_columbia.py --source datasets/external/columbia --overwrite
```

Универсальная команда:

```powershell
python scripts/02_import_dataset.py --kind columbia --source datasets/external/columbia --overwrite
python scripts/02_import_dataset.py --kind casia --source datasets/external/casia --overwrite
python scripts/02_import_dataset.py --kind comofod --source datasets/external/comofod --overwrite
python scripts/02_import_dataset.py --kind coverage --source datasets/external/coverage --overwrite
python scripts/02_import_dataset.py --kind tif-pairs --source datasets/external/image --overwrite
```

После импорта данные приводятся к единому виду:

```text
datasets/public/original
datasets/public/edited
datasets/public/manifest.csv
```

`manifest.csv` содержит `image_path`, `label`, `source_image`, `manipulation_type`, `dataset_name`.

### 2. Извлечь признаки

Основной режим обучения - только публичные/внешние данные:

```powershell
python scripts/03_extract_features.py --sources public
```

Результат:

```text
features/features.csv
```

На большом датасете скрипт печатает прогресс каждые 100 изображений. Для быстрой проверки можно обработать небольшой сэмпл:

```powershell
python scripts/03_extract_features.py --sources public --limit 1000
```

`label` нужен для обучения и оценки. Для нового изображения label не нужен: prediction-скрипт сам извлекает признаки и подает их в обученную модель.

### 3. Обучить Random Forest

Быстрый запуск без подбора гиперпараметров:

```powershell
python scripts/04_train_random_forest.py --scopes public --no-tune
```

Более долгий запуск с `GridSearchCV`:

```powershell
python scripts/04_train_random_forest.py --scopes public
```

Главная модель для дальнейшего predict:

```text
models/public_combined_random_forest.joblib
```

### 4. Собрать отчет по обучению

```powershell
python scripts/05_build_report.py
```

Результат:

```text
reports/course_report.md
reports/metrics.csv
reports/figures/
```

### 5. Проверить одно изображение

```powershell
python scripts/08_predict_image.py --image path\to\image.jpg --model models/public_combined_random_forest.joblib
```

Пример вывода:

```text
Prediction: edited
Probability original: 0.18
Probability edited: 0.82
Model: public_combined_random_forest.joblib
Saved prediction report to reports/predictions/latest_prediction.md
```

Отчет по одному изображению сохраняется отдельно:

```text
reports/predictions/latest_prediction.md
```

## Demo-режим

`generated` оставлен только для проверки пайплайна без внешних датасетов. Для финального результата курсовой лучше использовать `datasets/public`, импортированный из реальных forensic-датасетов.

```powershell
python scripts/run_pipeline.py --create-demo-source --demo-count 6 --no-tune --overwrite
```

## Что извлекается во features

Служебные колонки:

- `image_path`
- `source`
- `label`
- `label_name`
- `source_image`
- `manipulation_type`
- `dataset_name`

Они нужны для анализа и фильтрации, но не используются как входные признаки модели.

Входные признаки модели:

- `meta_*` - размер, формат, EXIF, Software, Make, Model, DateTime, Orientation, признаки потери metadata;
- `noise_*` - ELA, Laplacian variance, шумовой остаток, JPEG block artifacts, цветовые и частотные статистики.

Группы признаков для экспериментов:

- `metadata`
- `noise`
- `combined`

## Тесты

```powershell
python -m pytest
```

## Основные файлы

- `src/image_edit_detection/dataset.py` - импорт датасетов, generated demo, manifest, trimming.
- `src/image_edit_detection/features.py` - извлечение metadata/noise/ELA-признаков.
- `src/image_edit_detection/train.py` - обучение Random Forest, метрики, модели и графики.
- `src/image_edit_detection/predict.py` - prediction одного изображения.
- `src/image_edit_detection/report.py` - короткий отчет по результатам обучения.
- `scripts/08_predict_image.py` - CLI для проверки одного изображения.
