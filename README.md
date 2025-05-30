# Проект Frost Series

## Обзор

Проект Frost Series — это инициатива по анализу данных и моделированию, направленная на изучение деградации свойств материалов в условиях циклического замораживания. Он использует Python для обработки данных, статистического моделирования, машинного обучения и визуализации, чтобы анализировать экспериментальные данные и прогнозировать тенденции деградации. Кодовая база структурирована для обработки данных из Excel, выполнения анализа временных рядов, подгонки моделей деградации и генерации визуализаций для интерпретации.

## Структура проекта

Каталог проекта организован следующим образом:

- **data/**: Содержит необработанные файлы данных, такие как `Сводная ведомость.xlsx`, основной набор данных.
- **logs/**: Хранит файлы логов (например, `log.log`) для отладки и отслеживания выполнения.
- **scripts/**: Содержит основные скрипты Python для обработки данных и моделирования:
  - `data_loader.py`: Загружает и преобразует данные из Excel в структурированный DataFrame.
  - `holt_arima.py`: Реализует анализ временных рядов с использованием моделей ARIMA и Holt-Winters.
  - `math_model.py`: Определяет и подгоняет экспоненциально-логарифмическую модель деградации.
  - `ml.py`: Применяет методы машинного обучения (Random Forest, XGBoost) для прогнозирования.
  - `main.py`: Оркестрирует рабочий процесс, интегрируя загрузку данных, моделирование и визуализацию.
- **utils/**: Содержит вспомогательные модули:
  - `metrics.py`: Вычисляет метрики оценки (MAE, RMSE, MAPE).
  - `plotters.py`: Предоставляет функции для создания графиков деградации, свечей и точечных диаграмм.
  - `logger_config.py`: Настраивает функциональность логирования.
- `poetry.lock` и `pyproject.toml`: Файлы управления зависимостями для Poetry.

## Установка

Чтобы настроить проект локально, выполните следующие шаги:

1. **Клонируйте репозиторий:**
   ```bash
   git clone <repository-url>
   cd frost-series
   ```

2. **Установите зависимости:** Убедитесь, что Poetry установлен, затем выполните:
   ```bash
   poetry install
   ```

3. **Проверьте установку:** Активируйте виртуальное окружение и проверьте установленные пакеты:
   ```bash
   poetry shell
   pip list
   ```

## Использование

### Запуск проекта

Точка входа — `main.py`. Запустите его с помощью:
```bash
python main.py
```

Этот скрипт:
- Загружает данные из `data/Сводная ведомость.xlsx`.
- Подгоняет модель деградации с использованием метода наименьших квадратов (МНК).
- Выполняет прогнозирование временных рядов с помощью моделей ARIMA и Holt-Winters.
- Генерирует графики, сохраняемые по указанному пути (например, `C:\Users\tnick\Desktop`).

### Ключевые функции

- **`data_loader.py`:**
  - `load_and_transform`: Читает данные из Excel с многоуровневыми заголовками, преобразуя их в DataFrame в длинном формате с колонками для идентификаторов образцов, циклов замораживания и свойств (например, `rs`, `r`, `E50`).
  - `sort_params_for_ige`: Фильтрует данные по `ИГЭ` (например, `1a_t`) и вычисляет статистику (среднее, минимум, максимум) для параметра (например, `E50`) по циклам.

- **`holt_arima.py`:**
  - `interpolate_time_series`: Интерполирует неравномерные данные временных рядов.
  - `find_best_arima`: Оптимизирует параметры ARIMA на основе AIC.
  - `forecast_arima` и `forecast_holt`: Генерируют прогнозы с доверительными интервалами.

- **`math_model.py`:**
  - `degradation_model`: Определяет экспоненциально-логарифмическую функцию деградации.
  - `fit_degradation_model`: Подгоняет параметры модели с помощью подгонки кривых.

- **`ml.py`:**
  - Реализует инженерию признаков (лаги, разности, скользящие средние) и обучает модели Random Forest и XGBoost для прогнозирования.

- **`plotters.py`:**
  - `plot_degradation`: Строит графики тенденций деградации с тестовыми и прогнозными данными.
  - `plot_candlestick`: Создает графики с ошибками для статистики параметров.
  - `plot_scatter_with_mean`: Строит графики индивидуальных значений с средними трендами.

- **`metrics.py`:**
  - Вычисляет MAE, RMSE и MAPE для оценки производительности модели.

### Ввод данных

Проект ожидает файл Excel (`Сводная ведомость.xlsx`) с листом под названием `Переделка Лист 1`. Данные должны включать:
- Многоуровневые заголовки с идентификаторами образцов (например, `Лаб. № пробы`) и свойствами (например, `rs`, `r`, `E50`) по циклам замораживания (0, 3, 5, 10, 24).

### Вывод

- **Графики:** Сохраняются как PNG-файлы по указанному пути `save_path` (например, `C:\Users\tnick\Desktop`).
- **Логи:** Подробные логи выполнения в `logs/log.log`.
- **Метрики:** Выводятся в консоль, сравнивая прогнозы МНК, ARIMA и Holt с тестовыми данными.

## Зависимости

Проект зависит от следующих библиотек Python (управляемых через `pyproject.toml`):
- `pandas`: Манипуляция данными.
- `numpy`: Численные вычисления.
- `matplotlib`: Построение графиков.
- `statsmodels`: Модели ARIMA и Holt-Winters.
- `scikit-learn`: Random Forest.
- `xgboost`: Градиентный бустинг.
- `scipy`: Подгонка кривых.