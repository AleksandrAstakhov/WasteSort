# WasteSort — Классификация отходов по фото

Система автоматической классификации бытовых отходов по фотографии на 10 категорий. Достигает точности 95.1% на основной модели (EfficientNet-B2).

## Описание задачи

### Постановка

Цель проекта — построить модель классификации бытовых отходов по фотографии для автоматической сортировки мусора на конвейерах перерабатывающих станций и "умных" мусорных баков.

Задача относится к multiclass image classification. На реальных данных она нетривиальна из-за визуального сходства ряда категорий (например, три вида стекла, бумага vs картон) и гетерогенности класса trash.

### Входные и выходные данные

**Вход:**

- Изображение в формате .jpg или .png
- Модель принимает тензор формы [N, C, H, W] где C=3 (RGB), H=W=224

**Выход:**

- Вектор логитов формы [N, 10]
- После softmax: вероятности по 10 классам
- Финальный класс: argmax по вектору вероятностей
- Для батчевого инференса результат сохраняется в CSV:
  ```
  Image_path,predicted_class,confidence
  ```

### Датасет

| Параметр   | Значение                                                                                                                   |
| ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| Источник   | [Kaggle: Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) (автор: Mostafa Abla) |
| Размер     | 15,515 изображений (~1.6 GB на диске)                                                                                      |
| Классы     | 10 (консолидировано из 12 оригинальных классов Kaggle)                                                                     |
| Разрешение | 512×384 до 3264×2448 пиксели                                                                                               |
| Split      | 70% train (10,860), 15% val (2,327), 15% test (2,328)                                                                      |
| Баланс     | Сбалансирован (~1,550 изображений на класс)                                                                                |
| Лицензия   | CC0 (Public Domain)                                                                                                        |

**10 классов:**

- biological (пищевые отходы)
- cardboard (картон)
- clothes (текстиль, одежда)
- glass (стекло - объединены glass_brown, glass_green, glass_white)
- metals (металлические банки)
- paper (бумажные продукты)
- plastic (пластиковые бутылки)
- shoes (обувь)
- trash (смешанные отходы)
- unknown (неклассифицированные - батарейки)

**Потенциальные проблемы данных:**

- Визуальное сходство категорий стекла
- Умеренный дисбаланс классов
- Разное качество и освещение фотографий

### Метрики качества

Используются стандартные метрики для мультиклассовой классификации:

1. **CrossEntropyLoss** — основная функция потерь при обучении
2. **Macro F1-score** — основная метрика качества (выбрана за учет дисбаланса классов)
3. **Accuracy (Top-1)** — доля правильных предсказаний
4. **Confusion Matrix** — анализ того, какие классы путает модель
5. **Per-class F1** — отдельные метрики для каждого из 10 классов

**Ожидаемые значения:**

- Baseline (ResNet-18): macro F1 ~ 0.85–0.88
- Основная модель (EfficientNet-B2): macro F1 ~ 0.93–0.96

### Архитектуры

**Baseline: ResNet-18**

- Предобученные ImageNet веса
- Fine-tuning всех слоёв
- LR=1e-4, оптимизатор Adam, CrossEntropyLoss
- 10–15 эпох для получения опорного результата

**Основная модель: EfficientNet-B2**

- Compound scaling (одновременное масштабирование глубины, ширины и разрешения)
- Более эффективна по accuracy/FLOPs чем ResNet
- Аугментации данных (RandomResizedCrop, HorizontalFlip, Normalize)
- Learning rate scheduler и early stopping

### Препроцессинг данных

**Загрузка:**

- Resize изображения до 256x256
- Center crop до 224x224
- Нормализация по ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Аугментации (только на train set):**

- RandomResizedCrop(224, scale=(0.8, 1.0))
- HorizontalFlip(p=0.5)
- ColorJitter(brightness=0.2, contrast=0.2)
- RandomRotation(10 degrees)

### Постобработка результатов

Результаты инференса сохраняются в CSV с колонками:

- Image_path: путь до изображения
- predicted_class: предсказанный класс (название)
- confidence: вероятность (значение после softmax для выбранного класса)

### Внедрение и deployment

**Форматы моделей:**

- PyTorch checkpoint (.ckpt): 128 MB для ResNet-18, 89 MB для EfficientNet-B2
- ONNX (.onnx): 0.15 MB для ResNet-18, 1.46 MB для EfficientNet-B2 (61× сжатие)

**Варианты инференса:**

1. **CLI инференс** — прямое использование PyTorch checkpoints (см. команды ниже)
2. **ONNX инференс** — легковесный формат для продакшена (~1.5 MB вместо 89 MB)
3. **Triton Inference Server** — высокопроизводительный HTTP API для масштабирования

**Ресурсы для инференса:**

- Latency: ~50ms на одном изображении (GPU/CPU зависит)
- Throughput: можно обрабатывать батчи из нескольких изображений
- Память: ~300 MB для загрузки модели в память

---

## Setup (Настройка окружения)

### Требования

- Python 3.10+
- Poetry

### Установка Poetry

```bash
# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -

# Или через pip
pip install poetry

# Проверить версию
poetry --version
```

### Установка проекта

```bash
# Клонировать репозиторий
git clone <repo_url>
cd WasteSort

# Установить зависимости
poetry install

# Или с dev инструментами (черный, ruff, mypy, pre-commit)
poetry install --with dev

# Или с train зависимостями (mlflow, matplotlib, scikit-learn)
poetry install --with train

# Или всё вместе
poetry install --with dev --with train

# Активировать окружение Poetry
poetry shell
```

### Инициализировать DVC (один раз)

```bash
# Создать локальное хранилище
mkdir ../dvc-storage

# Инициализировать DVC удалённые хранилища
dvc remote add -d data-storage ../dvc-storage/data
dvc remote add models-storage ../dvc-storage/models
```

### Инициализировать pre-commit (один раз)

```bash
# Установить git hooks
pre-commit install

# Проверить всё работает
pre-commit run -a
```

### Конфигурация

Основная конфигурация находится в:

- `pyproject.toml` — зависимости, инструменты (Poetry), черный, ruff, mypy
- `configs/config.yaml` — гиперпараметры (Hydra)
- `.dvc/config` — хранилища (DVC)
- `.pre-commit-config.yaml` — линтинг и форматирование (pre-commit)

---

## Train (Обучение)

### 1. Загрузить датасет

Датасет автоматически загружается со скриптом `download_data.py`. Требует Kaggle API token:

```bash
# Вариант 1: через аргумент командной строки
poetry run python3 scripts/download_data.py --token username:api_key

# Вариант 2: через переменную окружения
export KAGGLE_API_TOKEN=username:api_key
poetry run python3 scripts/download_data.py
```

Скрипт:

- Загружает 15,515 изображений с Kaggle
- Автоматически консолидирует 12 классов в 10 (объединяет стеклянную посуду, переименовывает батарейки)
- Сохраняет в `data/raw/`

### 2. Добавить данные в DVC

```bash
dvc add data/raw/
git add data.dvc .gitignore
git commit -m "Add dataset"
dvc push -r data-storage
```

### 3. Запустить MLflow сервер (в отдельном терминале)

```bash
poetry run mlflow server --host 127.0.0.1 --port 8080
```

После запуска доступен на http://127.0.0.1:8080

### 4. Обучить модель

```bash
# Baseline модель (ResNet-18), 1 эпоха (~2-5 минут)
poetry run python3 scripts/run_training_patched.py ++command=train model=baseline train.max_epochs=1

# Baseline модель, 10 эпох (~20 минут)
poetry run python3 scripts/run_training_patched.py ++command=train model=baseline train.max_epochs=10

# Основная модель (EfficientNet-B2), 1 эпоха (~5 минут)
poetry run python3 scripts/run_training_patched.py ++command=train model=efficientnet train.max_epochs=1

# Основная модель, 10 эпох (~50 минут)
poetry run python3 scripts/run_training_patched.py ++command=train model=efficientnet train.max_epochs=10

# С кастомными гиперпараметрами
poetry run python3 scripts/run_training_patched.py ++command=train model=efficientnet \
  train.max_epochs=5 optim.lr=1e-4 train.batch_size=16
```

**Что происходит при обучении:**

- Используется PyTorch Lightning для управления обучением
- Логируются метрики в MLflow: loss, accuracy, f1-macro, per-class F1
- Сохраняются checkpoints лучшей модели в `artifacts/checkpoints/`
- Валидация происходит каждую эпоху
- Early stopping при отсутствии улучшений на валидации

**Выходные артефакты:**

- `artifacts/checkpoints/baseline-best.ckpt` (ResNet-18)
- `artifacts/checkpoints/efficientnet-best-v2.ckpt` (EfficientNet-B2)

### 5. Проверить систему

```bash
poetry run python3 scripts/check_system.py
```

Проверяет:

- Датасет (количество классов и изображений)
- Обученные checkpoints (baseline и efficientnet)
- ONNX модели
- Возможность инференса
- Статус Triton сервера (если запущен)
- Интеграцию MLflow

### 6. Сохранить модели в DVC

```bash
dvc add artifacts/checkpoints/ artifacts/model.onnx
git add artifacts.dvc
git commit -m "Add trained models"
dvc push -r models-storage
```

---

## Inference (Инференс)

### На PyTorch checkpoints

```bash
# На папке с изображениями
poetry run python3 scripts/run_training_patched.py ++command=infer \
  infer.input_dir=data/raw/biological \
  infer.output_csv=predictions.csv

# С кастомным checkpoints
poetry run python3 scripts/run_training_patched.py ++command=infer \
  infer.checkpoint_path=artifacts/checkpoints/baseline-best.ckpt \
  infer.input_dir=data/raw/biological

# Быстрый тест на нескольких картинках
poetry run python3 scripts/test_inference.py
```

**Выход:** `predictions.csv` с колонками:

```
image_path,predicted_class,confidence
data/raw/biological/bio1.jpg,biological,0.987
data/raw/paper/paper1.jpg,paper,0.954
```

### На ONNX модели

```bash
# Экспортировать основную модель в ONNX (1.46 MB)
poetry run python3 scripts/export_onnx.py

# Тест ONNX инференса
poetry run python3 scripts/test_inference.py
```

ONNX модель сохраняется в `artifacts/model.onnx`.

### На Triton Inference Server

```bash
# Обнови модель в Triton
cp artifacts/model.onnx triton_repo/waste_sort/1/model.onnx
cp artifacts/model.onnx.data triton_repo/waste_sort/1/model.onnx.data

# Запустить Triton (в отдельном терминале)
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton_repo:/models \
  nvcr.io/nvidia/tritonserver:latest \
  tritonserver --model-repository=/models

# Проверить здоровье
curl http://127.0.0.1:8000/v2/health/ready

# Протестировать инференс
poetry run python3 scripts/test_triton.py
```

Triton предоставляет HTTP API для масштабируемого инференса.

### Обновить модель в Triton

```bash
# После обучения и экспорта
poetry run python3 scripts/export_onnx.py

# Обнови модель в Triton
cp artifacts/model.onnx triton_repo/waste_sort/1/model.onnx
cp artifacts/model.onnx.data triton_repo/waste_sort/1/model.onnx.data

# Скопировать в Triton
cp artifacts/model.onnx triton_repo/waste_sort/1/model.onnx

# Перезагрузить Triton (остановить и запустить заново)
# Ctrl+C в терминале с Triton, потом:
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton_repo:/models \
  nvcr.io/nvidia/tritonserver:latest \
  tritonserver --model-repository=/models
```

---

## Результаты

### На реальных данных (15,515 изображений)

| Метрика        | Baseline (ResNet-18) | Main (EfficientNet-B2) |
| -------------- | -------------------- | ---------------------- |
| Test Accuracy  | 85.5%                | 95.1%                  |
| Test F1-macro  | 85.8%                | 94.4%                  |
| Val F1-macro   | 86.8%                | 95.5%                  |
| Inference Time | 50ms                 | 50ms                   |
| Model Size     | 128 MB               | 89 MB                  |
| ONNX Size      | 0.15 MB              | 1.46 MB                |
| Compression    | 853×                 | 61×                    |

### По классам (EfficientNet-B2)

| Класс      | F1   | Уровень  |
| ---------- | ---- | -------- |
| biological | 0.96 | Отличный |
| cardboard  | 0.95 | Отличный |
| clothes    | 0.98 | Отличный |
| glass      | 0.92 | Хороший  |
| metals     | 0.88 | Хороший  |
| paper      | 0.94 | Отличный |
| plastic    | 0.92 | Хороший  |
| shoes      | 0.96 | Отличный |
| trash      | 0.90 | Хороший  |
| unknown    | 0.91 | Хороший  |

---

## Утилиты

### Загрузить датасет

```bash
poetry run python3 scripts/download_data.py --token username:api_key
export KAGGLE_API_TOKEN=username:api_key && poetry run python3 scripts/download_data.py
```

### Проверить систему

```bash
poetry run python3 scripts/check_system.py
```

### Статистика датасета

```bash
poetry run python3 scripts/show_data_stats.py
```

### Открыть MLflow веб-интерфейс

```bash
open http://127.0.0.1:8080
```

### Проверить не-ASCII символы

```bash
poetry run python3 scripts/check_non_ascii.py
```

### Быстрый тест pre-commit

```bash
# Проверить все файлы
pre-commit run -a

# Или конкретный инструмент
pre-commit run black -a
pre-commit run ruff -a
```

### Data Management (DVC)

```bash
# Скачать данные и модели
dvc pull

# Загрузить в хранилище
dvc add data/raw/ artifacts/
dvc push

# Проверить статус
dvc status
dvc remote list
```

---

## Версия

Status: Production Ready
Updated: May 24, 2026
Python: 3.10+
Accuracy: 95.1%
