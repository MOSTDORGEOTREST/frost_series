import numpy as np
from typing import Tuple


def generate_soil_data(
    num_samples: int = 10000,
    N_real: np.ndarray = np.array([0, 3, 5, 10, 24, 35, 60]),
    E50_real: np.ndarray = np.array([23.91666667, 17.83333333, 8.21666667, 8.15, 7.83333333, 7.79, 7.61]),
    sample_params: np.ndarray = np.array([2.72, 1.95, 1.51, 44.4, 0.80, 29.1, 0.99, 57.4, 33.5, 23.9, -0.19])
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерация синтетических данных по физико-механическим параметрам грунта и модулям E50
    при различных циклах промораживания N.

    :param num_samples: количество сэмплов, которые нужно сгенерировать
    :param N_real: массив значений количества циклов промораживания
    :param E50_real: массив реальных значений E50 (МПа) для соответствующих N_real
    :param sample_params: массив эталонных физических параметров одного грунта
    :return: кортеж из:
             - массива физико-механических параметров (num_samples, num_features)
             - массива значений E50 (num_samples, len(N_real))
             - массива N_real (не изменяется)
    """

    # Стандартное отклонение для каждого физического параметра: 3% от абсолютного значения
    sigma_params: np.ndarray = 0.1 * np.abs(sample_params)

    # Стандартное отклонение для E50: ±0.3 МПа
    sigma_E50: float = 3

    # Списки для накопления результатов
    physical_params: list[np.ndarray] = []
    E50_data: list[np.ndarray] = []

    for _ in range(num_samples):
        # Вариации физических параметров
        params_sample: np.ndarray = sample_params + np.random.normal(loc=0.0, scale=sigma_params)
        physical_params.append(params_sample)

        # Вариации модуля E50
        E50_sample: np.ndarray = E50_real + np.random.normal(loc=0.0, scale=sigma_E50, size=len(N_real))
        E50_sample = np.clip(E50_sample, 5.0, 25.0)  # Ограничение: физически допустимый диапазон

        E50_data.append(E50_sample)

    # Преобразование в NumPy-массивы
    physical_array: np.ndarray = np.array(physical_params)
    E50_array: np.ndarray = np.array(E50_data)

    return physical_array, E50_array, N_real
