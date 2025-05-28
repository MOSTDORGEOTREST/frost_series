import numpy as np
from typing import Dict, List, Union
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler


def predict_E50(
    physical_params: Union[List[float], np.ndarray],
    initial_E50: Dict[float, float],
    model: Model,
    physical_scaler: MinMaxScaler,
    E50_scaler: MinMaxScaler,
    N_values: np.ndarray
) -> np.ndarray:
    """
    Прогноз значений E50 на основе физических параметров и корректирующих начальных значений.

    :param physical_params: список или массив физических параметров [rs, r, rd, n, e, W, Sr, WL, WP, Ip, IL]
    :param initial_E50: словарь известных значений E50, например {0: 23.9167, 3: 17.8333}
    :param model: обученная модель нейросети (например, TCN)
    :param physical_scaler: MinMaxScaler, обученный на физических параметрах
    :param E50_scaler: MinMaxScaler, обученный на выходных данных (E50)
    :param N_values: массив значений N, соответствующих столбцам E50
    :return: массив предсказанных (и откорректированных) значений E50 для каждого N
    """

    # Преобразование физических параметров к нужному виду и масштабирование
    physical_params = np.array(physical_params).reshape(1, -1)
    X_scaled = physical_scaler.transform(physical_params)
    X_model_input = X_scaled.reshape((1, 1, X_scaled.shape[1]))  # (samples, timesteps, features)

    # Прогноз модели
    E50_pred_scaled = model.predict(X_model_input, verbose=0)[0]
    E50_pred = E50_scaler.inverse_transform(E50_pred_scaled.reshape(1, -1)).flatten()

    # Корректировка по известным значениям
    for N, known_E50 in initial_E50.items():
        if N in N_values:
            index = np.where(N_values == N)[0][0]
            E50_pred[index] = known_E50

    return E50_pred
