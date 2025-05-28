import numpy as np
from pathlib import Path

from TCN.forecast import predict_E50
from TCN.model_loader import load_tcn_model
from utils.plotters import plot_E50_predictions

# Загрузка модели
root = Path(__file__).resolve().parent.parent
model_dir = root / "models" / "TCN_epochs50_batch32_lr0.0010"
model, physical_scaler, E50_scaler, N_values = load_tcn_model(model_dir)

# Исходные данные
real_N = np.array([0, 3, 5, 10, 24, 35, 60])
real_E50 = np.array([23.91666667, 17.83333333, 8.21666667, 8.15, 7.83333333, 7.79, 7.61])
sample_params = [2.72, 1.95, 1.51, 44.4, 0.80, 29.1, 0.99, 57.4, 33.5, 23.9, -0.19]
key_sets = [
    {0: 23.91666667},
    {0: 23.91666667, 3: 17.83333333},
    {0: 23.91666667, 3: 17.83333333, 5: 8.21666667},
    {0: 23.91666667, 3: 17.83333333, 5: 8.21666667, 10: 8.15},
    {0: 23.91666667, 3: 17.83333333, 5: 8.21666667, 10: 8.15, 24: 7.83333333}
]

# Предсказания для модели
predictions = []
for keys in key_sets:
    E50_pred = predict_E50(
        physical_params=sample_params,
        initial_E50=keys,
        model=model,
        physical_scaler=physical_scaler,
        E50_scaler=E50_scaler,
        N_values=N_values
    )
    predictions.append(E50_pred)

# Построение графиков
save_dir = root / "results"
plot_E50_predictions(
    sample_params=sample_params,
    real_N=real_N,
    real_E50=real_E50,
    key_sets=key_sets,
    N_values=N_values,
    predict_E50_fn=lambda params, keys: predict_E50(
        physical_params=params,
        initial_E50=keys,
        model=model,
        physical_scaler=physical_scaler,
        E50_scaler=E50_scaler,
        N_values=N_values
    ),
    save_path=save_dir
)