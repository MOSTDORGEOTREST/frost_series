"""
Модуль подгонки суперпозиционной модели деградации свойст грунтов при многократном промерзании.

Формула модели::

    p_norm(N) = p_inf + (1 - p_inf) * exp(-(lam * N)**gamma) - alpha * ln(1 + N)

где
    * ``p_inf``  – остаточное (стабилизировавшееся) значение свойства (0 < p_inf ≤ 1);
    * ``lam``    – коэффициент, аккумулирующий действие физических факторов (> 0);
    * ``gamma``  – показатель «резкости» начального спада (> 0);
    * ``alpha``  – коэффициент логарифмического накопления повреждений (>= 0).

"""
from typing import Tuple
import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "superposition_model",
    "fit_superposition_model",
]


def superposition_model(
        N: np.ndarray,
        p_inf: float,
        lam: float,
        gamma: float,
        alpha: float
) -> np.ndarray:
    """Возвращает значение нормированного параметра после N циклов.

    :param N: массив номеров циклов.
    :param p_inf: остаточное стабилизировавшееся значение.
    :param lam: интегральный коэффициент влияния физических факторов (>0).
    :param gamma: показатель резкости начального спада (>0).
    :param alpha: коэффициент логарифмической части (≥0).
    :return: массив p_norm той же формы, что и N.
    """
    return p_inf + (1.0 - p_inf) * np.exp(-np.power(lam * N, gamma)) - alpha * np.log1p(N)


def _normalize(y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Нормирует измеренные значения по первому элементу.

    :param y: массив абсолютных измерений свойства (shape (M,)).
    :return: кортеж (p_norm, y_max), где
             p_norm – нормированные значения,
             y_max – первое измеренное значение, принятое за 1.
    """
    if y.size == 0:
        raise ValueError("Пустой массив измерений.")
    y_max = float(y[0])
    if y_max <= 0:
        raise ValueError("Первое значение должно быть положительным.")
    return y / y_max, y_max


def fit_superposition_model(
        N: np.ndarray,
        y: np.ndarray,
        bounds: Tuple[Tuple[float, float, float, float],
                       Tuple[float, float, float, float]] | None = None,
        p0: Tuple[float, float, float, float] | None = None
        ) -> Tuple[Tuple[float, float, float, float], np.ndarray]:
    """Подбирает параметры суперпозиционной модели методом наименьших квадратов.

    :param N: массив номеров циклов (shape (M,)).
    :param y: массив измеренных абсолютных значений свойства (shape (M,)).
    :param bounds: границы параметров в формате (lower, upper); если *None* –
                   используются значения по умолчанию ((0,1e-8,1e-8,0), (1,10,10,1)).
    :param p0: начальное приближение параметров; если *None*, оценивается автоматически.
    :return: кортеж (params, p_norm_pred), где
             params – оценённые параметры (p_inf, lam, gamma, alpha),
             p_norm_pred – прогноз нормированных значений свойства на заданных *N*.
    :raises ValueError: если формы N и y не совпадают или «циклы» отрицательные.
    """
    if N.shape != y.shape:
        raise ValueError("Массивы N и y должны иметь одинаковую форму.")
    if np.any(N < 0):
        raise ValueError("Число циклов не может быть отрицательным.")

    p_norm, p_max = _normalize(y)

    if bounds is None:
        bounds = ((0.0, 1e-8, 1e-8, 0.0), (1.0, 10.0, 10.0, 1.0))

    if p0 is None:
        p0 = (float(p_norm[-1]),         # p_inf – примерно последнее значение
              1.0 / (N[-1] + 1e-6),      # lam   – порядок 1 / N_max
              1.0,                       # gamma – экспоненциальный режим
              0.01)                      # alpha – небольшой

    popt, _ = curve_fit(superposition_model, N, p_norm,
                        p0=p0, bounds=bounds, method="trf")

    return tuple(map(float, popt)), p_norm, p_max


if __name__ == "__main__":
    import argparse

    N = np.array([0, 3, 5, 10, 24])
    D = np.array([23.91666667, 17.83333333, 8.21666667, 8.15, 7.83333333])

    params, p_pred, p_max = fit_superposition_model(N, D)

    print("p_inf  = {:.6f}".format(params[0]))
    print("lam    = {:.6f}".format(params[1]))
    print("gamma  = {:.6f}".format(params[2]))
    print("alpha  = {:.6f}".format(params[3]))

    print("\n----- Прогноз p_norm -----")
    for n, p in zip(N.astype(int), p_pred):
        print(f"N={n:4d}: p_norm={p: .6f}")

    import matplotlib.pyplot as plt
    plt.scatter(N, D, c="b", marker="o")

    plt.plot(np.linspace(0, 30, 100), p_max * superposition_model(np.linspace(0, 30, 100), *params), color="b")
    plt.show()
