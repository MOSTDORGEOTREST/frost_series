import numpy as np
from typing import Tuple, Optional
from scipy.optimize import curve_fit
from functools import partial

def degradation_model(N: np.ndarray, A: float, B: float, C: float, D: float) -> np.ndarray:
    """
    Экспоненциально-логарифмическая модель деградации:

    D(N) = 1 - A * (1 - exp(-B * N)) + С * ln(N + 1)

    :param N: массив номеров циклов (целые >= 0)
    :param A: Начальное значение при 0 цикле
    :param B: вклад экспоненциальной фазы (>= 0)
    :param C: скорость насыщения экспоненты (>= 0)
    :param D: коэффициент логарифмической фазы (>= 0)
    :return: массив значений относительной деградации D(N)
    """
    return A - B * (1 - np.exp(-C * N)) - D * np.log1p(N + 1)

def simplified_degradation_model(N: np.ndarray, B: float, C: float, D: float) -> np.ndarray:
    return degradation_model(N, A=1.0, B=B, C=C, D=D)


def normalize_values(y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Нормализует значения по максимальному элементу:

    D_i = yi / y_max

    :param y: массив исходных значений свойства (y0, y1, ..., yM)
    :return: кортеж (D, y_max) — массив нормализованной деградации и начальное значение
    """
    y_max = np.max(y)
    if y_max == 0:
        raise ValueError("Первое значение y_max не может быть нулевым для нормализации.")
    D = y / y_max
    return D, y_max


def fit_degradation_model(
        N: np.ndarray,
        y: np.ndarray,
        A: Optional[bool] = True
) -> Tuple[Tuple[float, float, float], np.ndarray]:
    """
    Подбирает параметры модели деградации методом наименьших квадратов.

    :param N: массив номеров циклов (shape (M+1,))
    :param y: массив измеренных значений свойства после Ni циклов (shape (M+1,))
    :param normalize: если True, сначала нормализует y относительно y_max
    :return: кортеж (params, D_norm), где
             params = (A_est, B_est, C_est),
             D_norm = массив нормализованных значений D_i
    """
    # Проверка формата входа
    if N.shape != y.shape:
        raise ValueError("Массивы N и y должны иметь одинаковую форму.")

    # Нормализация
    D, y_max = normalize_values(y)

    # Границы параметров: все неотрицательные
    bounds = (0, 3)

    # Начальная догадка:
    #   B0 — размах экспоненциальной фазы ≈ D[-1]
    #   C0 — скорость насыщения ≈ 1 / (последний N)
    #   D0 — вклад логарифмической фазы ≈ 10% от D[-1]
    A0 = float(np.max(D))
    B0 = float(np.min(D))
    C0 = 1
    D0 = 0.00001

    if A:
        p0 = [A0, B0, C0, D0]

        popt, _ = curve_fit(
            f=degradation_model,
            xdata=N,
            ydata=D,
            p0=p0,
            bounds=bounds,
            method='trf'
        )

        A_est, B_est, C_est, D_est = popt
    else:
        B0 = float(np.min(D))
        C0 = 1
        D0 = 0.00001
        p0 = [B0, C0, D0]

        # Подгонка модели четырёх параметров
        popt, _ = curve_fit(
            f=simplified_degradation_model,
            xdata=N,
            ydata=D,
            p0=p0,
            bounds=bounds,
            method='trf'
        )

        # Распаковываем все четыре параметра
        B_est, C_est, D_est = popt
        A_est = 1.0

    return (A_est, B_est, C_est, D_est), D
