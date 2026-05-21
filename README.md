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

```powershell
git clone https://github.com/KorolDz/ML
cd H:\ML
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


### 4. Проверить одно изображение

```powershell
python scripts/08_predict_image.py --image path\to\image.jpg --model models/public_combined_random_forest.joblib
```

Отчет по одному изображению сохраняется отдельно:

```text
reports/predictions/latest_prediction.md
reports/predictions/latest_prediction_evidence.png
```

По умолчанию рядом с отчетом создается ELA-визуализация: оригинал, ELA-карта и красная псевдомаска подозрительных областей. Это не ground-truth маска редактирования, а наглядная forensic-подсказка. Полезные параметры:

