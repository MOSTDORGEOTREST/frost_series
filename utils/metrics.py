import numpy as np
from typing import Dict, Tuple


def round_two_digits_after_point(x: float) -> float:
    """
    Округляет число до двух значащих цифр **после запятой**.

    :param x: Входное число
    :return: Округлённое число (две значащих после запятой)
    """
    str_x = f"{x:.20f}"  # безопасно вывести с большим числом знаков
    int_part, frac_part = str_x.split(".")

    # Убираем лишние нули справа и считаем первые две значащие цифры в дробной части
    result_frac = ""
    significant_count = 0
    for ch in frac_part:
        result_frac += ch
        if ch != '0':
            significant_count += 1
        if significant_count == 2:
            break

    # Добавляем недостающие нули, чтобы было корректно округлено
    needed_len = len(result_frac)
    rounded = round(x, needed_len)
    return rounded


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисление средней абсолютной ошибки (MAE).

    :param y_true: истинные значения
    :param y_pred: прогнозные значения
    :return: MAE
    """
    err = np.mean(np.abs(y_true - y_pred))
    return err


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисление корня из среднеквадратичной ошибки (RMSE).

    :param y_true: истинные значения
    :param y_pred: прогнозные значения
    :return: RMSE
    """
    err = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return err

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Средняя абсолютная процентная ошибка (%).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0  # избегаем деления на ноль
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_models(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Оценка нескольких моделей по метрикам MAE и RMSE.

    :param predictions: словарь с предсказаниями, где ключ — название модели,
                        а значение — кортеж (N_pred: np.ndarray, D_pred: np.ndarray)
    :param y_true: истинные значения для сравнения (в том же порядке)
    :return: словарь метрик для каждой модели
    """
    results: Dict[str, Dict[str, float]] = {}
    for name, (_, d_pred) in predictions.items():
        if len(d_pred) != len(y_true):
            raise ValueError(
                f"Length mismatch for model '{name}': true {len(y_true)}, pred {len(d_pred)}"
            )
        n = min(len(y_true), len(d_pred))
        y_t = y_true[:n]
        y_p = d_pred[:n]

        _mae = mae(y_t, y_p)
        _rmse = rmse(y_t, y_p)
        _mape = mape(y_t, y_p)

        results[name] = {
            'MAE': round_two_digits_after_point(_mae),
            'RMSE': round_two_digits_after_point(_rmse),
            'MAPE': round_two_digits_after_point(_mape),
        }
    return results

