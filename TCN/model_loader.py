from tensorflow.keras.models import load_model
import os
import numpy as np
from typing import Tuple
from tensorflow.keras.models import Sequential, save_model
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_tcn_model(model_folder: str) -> Tuple[Sequential, MinMaxScaler, MinMaxScaler, np.ndarray]:
    """
    Загружает модель, два скейлера и массив N_values из заданной папки.

    :param model_folder: путь к папке, где лежат model.keras, *.pkl и N_values.npy
    :return: модель, скейлер физических параметров, скейлер E50, массив N_values
    """
    model = load_model(os.path.join(model_folder, "model.keras"))
    physical_scaler = joblib.load(os.path.join(model_folder, "physical_scaler.pkl"))
    E50_scaler = joblib.load(os.path.join(model_folder, "E50_scaler.pkl"))
    N_values = np.load(os.path.join(model_folder, "N_values.npy"))

    return model, physical_scaler, E50_scaler, N_values
