import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing, HoltWintersResults
from typing import List, Tuple, Optional, Dict, Any
import itertools
from logger_config import logger

warnings.filterwarnings("ignore")

def interpolate_time_series(time_stamps: List[int], values: List[float], freq: int = 1) -> Tuple[pd.Series, np.ndarray, pd.DataFrame]:
    """
    Линейная интерполяция значений временного ряда на равномерной сетке.

    :param time_stamps: список временных меток (неравномерные точки во времени)
    :param values: значения на соответствующих временных метках
    :param freq: шаг между равномерными временными точками
    :return: кортеж из:
             - интерполированного ряда (pd.Series)
             - массива временных точек (np.ndarray)
             - DataFrame с исходными и интерполированными значениями
    """
    time_range = np.arange(time_stamps[0], time_stamps[-1] + 1, freq)
    df = pd.DataFrame({'time': time_stamps, 'value': values})
    df_interp = pd.DataFrame({'time': time_range})
    df_interp = df_interp.merge(df, on='time', how='left')
    interpolated = df_interp['value'].interpolate(method='linear').clip(lower=0)
    df_interp['value'] = interpolated
    return interpolated, time_range, df_interp


def test_stationarity(series: pd.Series, label: str = "") -> float:
    """
    Проверка временного ряда на стационарность с помощью теста Дики-Фуллера.

    :param series: одномерный временной ряд (pd.Series)
    :param label: описание ряда (для логгера)
    :return: p-value теста (чем меньше, тем выше вероятность стационарности)
    """
    result = adfuller(series, autolag='AIC')
    logger.info(f"{label} ADF statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        logger.info(f"{label} Ряд стационарен (p <= 0.05)")
    else:
        logger.info(f"{label} Ряд нестационарен (p > 0.05)")
    return result[1]



def find_best_arima(series: pd.Series, p_range: range, d_range: range, q_range: range) -> Tuple[Optional[ARIMAResults], Optional[Tuple[int, int, int]], float]:
    """
    Подбор наилучшей модели ARIMA на основе AIC.

    :param series: временной ряд
    :param p_range: диапазон значений p (AR)
    :param d_range: диапазон значений d (интеграция)
    :param q_range: диапазон значений q (MA)
    :return: кортеж из:
             - обученной модели ARIMA (или None, если не удалось)
             - параметров (p, d, q)
             - значения AIC
    """
    best_model = None
    best_pdq = None
    best_aic = float("inf")

    for pdq in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(series, order=pdq)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_model = model_fit
                best_pdq = pdq
                best_aic = model_fit.aic
            #logger.info(f"ARIMA{pdq} AIC: {model_fit.aic:.2f}")
        except Exception as e:
            continue

    return best_model, best_pdq, best_aic


def forecast_arima(model: ARIMAResults, steps: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Прогнозирование временного ряда с помощью обученной модели ARIMA.

    :param model: обученная модель ARIMA
    :param steps: число шагов вперёд для прогноза
    :return: кортеж из:
             - значений прогноза
             - доверительных интервалов (DataFrame с колонками нижнего и верхнего доверительного интервала)
    """
    forecast_obj = model.get_forecast(steps=steps)
    forecast_values = forecast_obj.predicted_mean.values
    conf_int = forecast_obj.conf_int()
    return forecast_values, conf_int


def forecast_holt(series: pd.Series, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Прогнозирование временного ряда методом экспоненциального сглаживания Хольта.

    :param series: временной ряд
    :param steps: количество шагов вперёд
    :return: кортеж из:
             - прогнозных значений
             - нижней границы 95% ДИ
             - верхней границы 95% ДИ
    """
    model = ExponentialSmoothing(series, trend='add', seasonal=None, damped_trend=True)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=steps)
    residuals = model_fit.fittedvalues - series
    sigma = residuals.std(ddof=0)
    z = 1.96

    lower = forecast - z * sigma
    upper = forecast + z * sigma

    return forecast.values, lower.values, upper.values
