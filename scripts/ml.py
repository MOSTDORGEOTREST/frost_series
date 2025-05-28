import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

from logger_config import logger

# Данные
data = [9, 8, 6, 2.5, 2, 1.9]
time_stamps = [0, 3, 5, 10, 15, 24]

# Создаём DataFrame
df = pd.DataFrame({'time': time_stamps, 'value': data})

# 1. Создание признаков
# Лаги: предыдущие значения
df['lag_1'] = df['value'].shift(1)
df['lag_2'] = df['value'].shift(2)

# Разности первого порядка
df['diff_1'] = df['value'].diff()

# Временные признаки
df['time_squared'] = df['time'] ** 2
df['time_log'] = np.log1p(df['time'])

# Новый признак: экспоненциальное затухание
df['exp_decay'] = np.exp(-0.1 * df['time'])  # Затухание с коэффициентом 0.1

# Новый признак: скользящее среднее
df['rolling_mean'] = df['value'].rolling(window=2).mean()

# Удаляем строки с NaN (из-за лагов и скользящего среднего)
df = df.dropna()

# Признаки и целевая переменная
X = df[['time', 'lag_1', 'lag_2', 'diff_1', 'time_squared', 'time_log', 'exp_decay', 'rolling_mean']]
y = df['value']

# 2. Обучение Random Forest
logger.info("Обучение Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_split=2, random_state=42)
rf_model.fit(X, y)

# 3. Обучение XGBoost
logger.info("Обучение XGBoost...")
xgb_model = XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, reg_lambda=0.5, random_state=42)
xgb_model.fit(X, y)

# 4. Прогноз на 5 шагов вперёд
n_forecast = 5
forecast_rf = []
forecast_xgb = []
last_row = df.iloc[-1].copy()

# Временные метки для прогноза (шаг 9, как последний интервал 24-15)
future_times = [last_row['time'] + (i + 1) * (last_row['time'] - df.iloc[-2]['time']) for i in range(n_forecast)]

for i in range(n_forecast):
    # Подготовка признаков для следующей точки
    next_time = future_times[i]
    next_features = pd.DataFrame({
        'time': [next_time],
        'lag_1': [last_row['value']],
        'lag_2': [last_row['lag_1']],
        'diff_1': [last_row['value'] - last_row['lag_1']],
        'time_squared': [next_time ** 2],
        'time_log': [np.log1p(next_time)],
        'exp_decay': [np.exp(-0.1 * next_time)],
        'rolling_mean': [(last_row['value'] + last_row['lag_1']) / 2]
    })

    # Прогноз Random Forest
    rf_pred = rf_model.predict(next_features)[0]
    forecast_rf.append(rf_pred)

    # Прогноз XGBoost
    xgb_pred = xgb_model.predict(next_features)[0]
    forecast_xgb.append(xgb_pred)

    # Обновляем last_row для следующей итерации
    last_row['time'] = next_time
    last_row['lag_2'] = last_row['lag_1']
    last_row['lag_1'] = last_row['value']
    last_row['value'] = xgb_pred  # Используем XGBoost для последовательного обновления

# Ограничим прогнозы, чтобы они не превышали ожидаемый диапазон (1.8–2.0)
forecast_rf = np.clip(forecast_rf, 1.8, 2.0)
forecast_xgb = np.clip(forecast_xgb, 1.8, 2.0)

# Логирование прогнозов
logger.info(f"Прогноз Random Forest на {n_forecast} шагов вперед: {forecast_rf.tolist()}")
logger.info(f"Прогноз XGBoost на {n_forecast} шагов вперед: {forecast_xgb.tolist()}")

# 5. Модель Хольта (для сравнения)
# Интерполяция для Хольта
time_range = np.arange(0, 25, 1)
df_interpolated = pd.DataFrame({'time': time_range})
df_interpolated = df_interpolated.merge(df[['time', 'value']], on='time', how='left')
time_series = df_interpolated['value'].interpolate(method='linear')
time_series[time_stamps[-1]:] = time_series[time_stamps[-1]]

logger.info("Обучение модели Хольта...")
try:
    model_holt = ExponentialSmoothing(time_series, trend='add', seasonal=None, damped_trend=True)
    model_holt_fit = model_holt.fit()
    forecast_holt = model_holt_fit.forecast(steps=n_forecast)
    logger.info(f"Прогноз Хольта на {n_forecast} шагов вперед: {forecast_holt.tolist()}")
except Exception as e:
    logger.error(f"Ошибка при обучении модели Хольта: {e}")
    forecast_holt = None

# 6. Визуализация прогнозов
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['value'], 'bo-', label='Исходные данные')
plt.plot(future_times, forecast_rf, label='Прогноз Random Forest', color='orange')
plt.plot(future_times, forecast_xgb, label='Прогноз XGBoost', color='red')
if forecast_holt is not None:
    plt.plot(range(int(time_range[-1]) + 1, int(time_range[-1]) + 1 + n_forecast), forecast_holt, label='Прогноз Хольта', color='green')
plt.legend()
plt.title('Прогноз временного ряда: Random Forest vs XGBoost vs Holt')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.show()