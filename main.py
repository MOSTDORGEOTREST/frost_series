import numpy as np
from logger_config import logger
from pprint import pformat

# 📦 Импорт модели деградации и процедуры подбора параметров
from scripts.math_model import (
    fit_degradation_model,         # МНК-подбор параметров A, B, C, D
    degradation_model              # Вычисление модели деградации по параметрам
)

# 📈 Импорт функций анализа временных рядов
from scripts.holt_arima import (
    interpolate_time_series,
    find_best_arima,
    forecast_arima,
    forecast_holt,
)

# 📊 Импорт функции визуализации
from utils.plotters import plot_degradation

# 📊 Импорт метрик
from utils.metrics import evaluate_models



# 🧪 Входные данные: измеренные значения деградации
#D = np.array([10, 8, 6, 2.5, 2, 1.9])
#N = np.array([0, 3, 5, 10, 15, 24])
N = np.array([0, 3, 5, 10, 24])
D = np.array([23.91666667, 17.83333333, 8.21666667, 8.15, 7.83333333])

# 🧪 Измеренные значения деградации для проверки прогноза
#D_additional = np.array([1.87, 1.83])
#N_additional = np.array([35, 60])
D_additional = np.array([7.79, 7.61])
N_additional = np.array([35, 60])
D_additional_norm = D_additional / np.max(D)

forecast_cycles_count = int(60 - np.max(N))


# ⚙️ Подбор параметров модели
logger.info("▶ Начинается подбор параметров МКЭ")
(A_est, B_est, C_est, D_est), D_norm = fit_degradation_model(
    N=N,
    y=D,
    A=False
)
logger.success(f"✅ Подбор параметров МКЭ завершен. A={A_est}, B={B_est}, C={C_est}, D={D_est}")

#A_est, B_est, C_est, D_est = 1.0, 0.657, 0.574, 0.0087


logger.info("▶ Построение прогноза ARIMA...")
# 📈 Интерполяция и подготовка временного ряда
interp_series, time_uniform, df_interp = interpolate_time_series(N.tolist(), D_norm.tolist(), freq=1)

# 📈 Прогноз с помощью ARIMA
arima_model, pdq, aic = find_best_arima(interp_series, p_range=range(5), d_range=range(3), q_range=range(5))

if arima_model:
    arima_forecast, arima_conf_int = forecast_arima(arima_model, steps=20)
    logger.success(f"✅ Прогноз ARIMA успешно построен.")
else:
    logger.warning("❌ Не удалось подобрать модель ARIMA.")

arima_forecast, arima_conf_int = forecast_arima(arima_model, steps=forecast_cycles_count)

logger.info("▶ Построение прогноза Хольта...")
# 📉 Прогноз с помощью метода Хольта
holt_forecast, holt_lower, holt_upper = forecast_holt(interp_series, steps=forecast_cycles_count)
logger.success("✅ Прогноз Хольта успешно построен.")


last_n = N[-1]
forecast_n = np.arange(last_n + 1, last_n + 1 + forecast_cycles_count)
last_value = interp_series.iloc[-1]


# Последний цикл из исходных данных
last_n = N[-1]
indices = (N_additional - (last_n + 1)).astype(int)

# Формируем словарь с прогнозами
predictions = {
    'LSM':   (N_additional, degradation_model(N_additional, A=A_est, B=B_est, C=C_est, D=D_est)),
    'ARIMA': (N_additional, arima_forecast[indices]),
    'HOLT':  (N_additional, holt_forecast[indices]),
}

metrics = evaluate_models(predictions, D_additional_norm)

logger.info("Результаты обработки данных:\n{}", pformat(metrics, width=80, compact=True))

plot_degradation(
    parameters_dict={
        'test_data': {
            'N': N,
            'D': D_norm,
        },
        'additional_test_data': {
            'N': N_additional.tolist(),
            'D': D_additional_norm.tolist(),
        },
        'LSM': {
            'N': np.linspace(0, last_n + forecast_cycles_count, 100),
            'D': degradation_model(np.linspace(0, last_n + forecast_cycles_count, 100), A=A_est, B=B_est, C=C_est, D=D_est),
        },
        'ARIMA': {
            'N': forecast_n.tolist(),
            'D': arima_forecast.tolist(),
        },
        'HOLT': {
            'N': forecast_n.tolist(),
            'D': holt_forecast.tolist(),
        },
    },
    title='Деградация параметра E50',
    ylabel='E50',
    metrics=metrics,
    save_path=r'results/'
)
