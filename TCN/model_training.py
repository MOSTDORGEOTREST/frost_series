import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib  # Для сохранения скейлеров


def train_tcn_model(
    physical_df: pd.DataFrame,
    E50_df: pd.DataFrame,
    save_directory: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[Sequential, MinMaxScaler, MinMaxScaler, np.ndarray]:
    """
    Обучает сверточную нейросеть (TCN) и сохраняет модель и скейлеры в отдельную папку.

    :param physical_df: DataFrame с физическими параметрами грунта
    :param E50_df: DataFrame с целевыми значениями E50
    :param save_directory: директория, в которой будет создана подпапка для модели
    :param epochs: количество эпох обучения
    :param batch_size: размер батча
    :param learning_rate: learning rate
    :return: модель, скейлеры, массив значений N
    """

    # Масштабируем данные
    physical_scaler = MinMaxScaler()
    E50_scaler = MinMaxScaler()

    X_scaled = physical_scaler.fit_transform(physical_df)
    y_scaled = E50_scaler.fit_transform(E50_df)

    X = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    y = y_scaled

    # Создание модели
    model = Sequential([
        Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        Conv1D(filters=32, kernel_size=1, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(y.shape[1])
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    # Обучение
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    # Сохранение
    N_values = E50_df.columns.to_numpy(dtype=float)
    if save_directory is not None:
        model_subdir = f"TCN_epochs{epochs}_batch{batch_size}_lr{learning_rate:.4f}"
        full_path = os.path.join(save_directory, model_subdir)
        os.makedirs(full_path, exist_ok=True)

        # Сохраняем модель и скейлеры
        save_model(model, os.path.join(full_path, "model.keras"))
        joblib.dump(physical_scaler, os.path.join(full_path, "physical_scaler.pkl"))
        joblib.dump(E50_scaler, os.path.join(full_path, "E50_scaler.pkl"))
        np.save(os.path.join(full_path, "N_values.npy"), N_values)

        print(f"Модель и скейлеры сохранены в папке: {full_path}")

    return model, physical_scaler, E50_scaler, N_values


if __name__ == '__main__':
    from pathlib import Path
    from TCN.data_preparation import generate_soil_data

    # Генерация данных
    physical_params, E50_data, N_values = generate_soil_data()

    # Преобразование в DataFrame
    physical_df = pd.DataFrame(physical_params)
    E50_df = pd.DataFrame(E50_data, columns=N_values)

    root = Path(__file__).resolve().parent.parent
    save_dir = root / "models"

    model, physical_scaler, E50_scaler, N_values = train_tcn_model(
        physical_df=physical_df,
        E50_df=E50_df,
        save_directory=save_dir,
        epochs=50,
        batch_size=32,
        learning_rate=0.001
    )