# Информация про студента
Король Денис Андреевич  
Korol_DA_23  
3 курс 6 семестр  
Специальность: Кибербезопасность  
Курсовой проект  

# Forensic-анализ редактирования изображений

Курсовой проект по машинному обучению в направлении **"Обнаружение фейков и цифровых манипуляций в медиафайлах"**.

Выбранная подтема: **анализ метаданных и шумовых паттернов изображений для определения следов редактирования в графических редакторах**.

Задача проекта — обучить модель бинарной классификации `original` / `edited` на forensic-признаках изображения: metadata, EXIF, Error Level Analysis, шумовой остаток и JPEG-артефакты. Основной метод — **Random Forests / Случайные леса**.

Проект не решает задачи GAN/diffusion detection, DeepFake-видео или NLP fake-news: здесь используется классическое признаковое описание изображения и табличная ML-модель.

## 🛠 Требования

- Python 3.9+
- pip

## 🚀 Как запустить

1. **Клонируйте репозиторий**  
   ```bash
   git clone https://github.com/KorolDz/ML
   cd ML

2. **Создайте и активируйте виртуальное окружение**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1 

3. **Установите зависимости**
   ```bsh
   pip install -r requirements.txt
  
4. **Подготовьте датасет**
   Скачайте один из forensic-датасетов (например, Columbia Uncompressed).
   Распакуйте в папку datasets/external/columbia (должны быть подпапки 4cam_auth и 4cam_splc).
   Импортируйте датасет в единую структуру:
   ```bash
   python scripts/02_import_dataset.py --kind columbia --source datasets/external/columbia --overwrite

5. **Извлеките признаки**
   ```bash
   python scripts/03_extract_features.py --sources public

6. **Обучите модель Random Forest**
   Быстрый запуск (без подбора гиперпараметров):
   ```bash
   python scripts/04_train_random_forest.py --scopes public --no-tune
   С подбором гиперпараметров (GridSearchCV):
   ```bash
   python scripts/04_train_random_forest.py --scopes public

7. Проверьте одно изображение
   ```bash
   python scripts/08_predict_image.py --image путь/к/изображению.jpg






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
- **MISD**  
  Структура `Au`, `Sp`, `Ground Truth Masks`. Импортируется через `--kind misd`: `Au` -> `original`, `Sp` -> `edited`, маски игнорируются.
- **IMD2020 Real-Life Manipulated Images**  
  Импортируется через `--kind imd2020`: файлы `*_orig.*` -> `original`, остальные изображения без `_mask` -> `edited`.
- **IMD2020 Large-Scale Set of Real Images**  
  Импортируется через `--kind imd2020-real`: все изображения добавляются только в `original`.
- **IMD2020 Large-Scale Set of Inpainting Images**  
  Импортируется через `--kind imd2020-inpainting`: все изображения добавляются только в `edited`, `manipulation_type=inpainting`.
- **Realistic Tampering Dataset**  
  Импортируется через `--kind realistic-tampering`: `pristine` -> `original`, `tampered-realistic` -> `edited`, `ground-truth` игнорируется.

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
  Dataset/Dataset/       # MISD: Au, Sp, Ground Truth Masks
  IMD2020/               # Real-Life Manipulated Images
  IMD2020_real_01/       # Large-Scale real images, part 1
  IMD2020_real_02/
  IMD2020_real_03/
  IMD2020_Generative_Image_Inpainting_yu2018_01/
  IMD2020_Generative_Image_Inpainting_yu2018_02/
  IMD2020_Generative_Image_Inpainting_yu2018_03/
  IMD2020_Generative_Image_Inpainting_yu2018_04/
  IMD2020_Generative_Image_Inpainting_yu2018_05/
  realistic-tampering-dataset/
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
python scripts/02_import_dataset.py --kind misd --source datasets/external/Dataset/Dataset --overwrite
python scripts/02_import_dataset.py --kind imd2020 --source datasets/external/IMD2020 --overwrite
python scripts/02_import_dataset.py --kind imd2020-real --source datasets/external/IMD2020_real_01 --overwrite
python scripts/02_import_dataset.py --kind imd2020-real --source datasets/external/IMD2020_real_02 --overwrite
python scripts/02_import_dataset.py --kind imd2020-real --source datasets/external/IMD2020_real_03 --overwrite
python scripts/02_import_dataset.py --kind imd2020-inpainting --source datasets/external/IMD2020_Generative_Image_Inpainting_yu2018_01 --overwrite
python scripts/02_import_dataset.py --kind imd2020-inpainting --source datasets/external/IMD2020_Generative_Image_Inpainting_yu2018_02 --overwrite
python scripts/02_import_dataset.py --kind imd2020-inpainting --source datasets/external/IMD2020_Generative_Image_Inpainting_yu2018_03 --overwrite
python scripts/02_import_dataset.py --kind imd2020-inpainting --source datasets/external/IMD2020_Generative_Image_Inpainting_yu2018_04 --overwrite
python scripts/02_import_dataset.py --kind imd2020-inpainting --source datasets/external/IMD2020_Generative_Image_Inpainting_yu2018_05 --overwrite
python scripts/02_import_dataset.py --kind realistic-tampering --source datasets/external/realistic-tampering-dataset --overwrite
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
Saved ELA evidence visualization to reports/predictions/latest_prediction_evidence.png
Saved prediction report to reports/predictions/latest_prediction.md
```

Отчет по одному изображению сохраняется отдельно:

```text
reports/predictions/latest_prediction.md
reports/predictions/latest_prediction_evidence.png
```

По умолчанию рядом с отчетом создается ELA-визуализация: оригинал, ELA-карта и красная псевдомаска подозрительных областей. Это не ground-truth маска редактирования, а наглядная forensic-подсказка. Полезные параметры:

```powershell
python scripts/08_predict_image.py --image path\to\image.jpg --no-visualization
python scripts/08_predict_image.py --image path\to\image.jpg --visualization-output reports\predictions\evidence.png --ela-quality 90 --ela-threshold-percentile 95
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
