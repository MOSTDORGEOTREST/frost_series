import sys
import os
import traceback
from typing import Dict, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
import pandas as pd

try:
    from scipy.optimize import curve_fit  # type: ignore
except ImportError:
    curve_fit = None

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.edgecolor": "black",
    "axes.grid": True,
    "figure.subplot.left": 0.15,
    "figure.subplot.right": 0.95,
    "figure.subplot.bottom": 0.15,
    "figure.subplot.top": 0.9,
})

def generate_normal_array(
    mean: Union[int, float],
    sigma: Union[int, float],
    n_points: int,
    accuracy_pct: Union[int, float] = 0.5
) -> np.ndarray:
    """
    Формирует выборку N(mean, sigma²) длиной ``n_points`` так,
    чтобы её среднее ⟨x⟩ отличалось от заданного ``mean``
    не более чем на ``accuracy_pct`` % (|⟨x⟩ − mean| ≤ accuracy_pct·mean/100).

    :param mean: требуемое математическое ожидание (μ) распределения
    :param sigma: стандартное отклонение (σ) распределения; σ ≥ 0
    :param n_points: количество генерируемых значений; n_points > 0
    :param accuracy_pct: допустимое процентное отклонение среднего
                         (по умолчанию 0.5 %)
    :return: ndarray с длиной ``n_points`` и заданной точностью среднего
    :raises ValueError: если входные аргументы вне диапазона допустимых значений
    """
    if n_points <= 0:
        raise ValueError("n_points должен быть положительным целым числом")
    if sigma < 0:
        raise ValueError("sigma должно быть неотрицательным")
    if accuracy_pct < 0:
        raise ValueError("accuracy_pct не может быть отрицательным")

    rng = np.random.default_rng()
    tolerance = abs(mean) * (accuracy_pct / 100.0)

    for _ in range(10):                                # до 10 попыток корректировки
        sample = rng.normal(loc=mean, scale=sigma, size=n_points)
        sample += mean - sample.mean()                 # сдвиг к требуемому mean
        if abs(sample.mean() - mean) <= tolerance:
            return sample

    # Форсированная коррекция (крайне редко понадобится)
    return sample - (sample.mean() - mean)


def load_and_transform(file_path: str) -> pd.DataFrame:
    """
    Загружает Excel-файл с двумя строками заголовков,
    преобразует в таблицу с колонками:
    sample_id, freezing_cycles и все параметры в отдельных столбцах.
    """
    # Получаем готовый DataFrame
    df = pd.read_excel(
        file_path,
        sheet_name='Лист1',
        header=0
    )

    # Список свойств
    properties = ['Е50', 'Eoed', 'Eur', 'Cu']

    # Уникальные столбцы
    id_columns = ['№', 'ИГЭ']

    # Циклы и соответствующие суффиксы
    cycles = [0, 10]
    suffixes = ['-0', '-10']

    # Список для хранения данных по каждому циклу
    dfs = []

    # Преобразование для каждого цикла
    for cycle, suffix in zip(cycles, suffixes):
        # Столбцы для текущего цикла
        cycle_columns = [prop + suffix for prop in properties]
        # Временный DataFrame для этого цикла
        temp_df = df[id_columns + cycle_columns].copy()
        # Переименовываем столбцы, убирая суффиксы
        temp_df.columns = id_columns + properties
        # Добавляем столбец 'Цикл промораживания'
        temp_df['Цикл промораживания'] = cycle
        dfs.append(temp_df)

    # Объединяем все DataFrame
    long_df = pd.concat(dfs, ignore_index=True)

    # Сортируем по '№' и 'Цикл промораживания'
    long_df = long_df.sort_values(by=['№', 'Цикл промораживания']).reset_index(drop=True)

    return long_df


def superposition_model(N: np.ndarray, p_inf: float, lam: float, gamma: float, alpha: float) -> np.ndarray:  # noqa: N803
    """p(N) = p_inf + (1−p_inf)·exp[−(λN)^γ] − α·ln(1+N)."""
    return p_inf + (1.0 - p_inf) * np.exp(-(lam * N) ** gamma) - alpha * np.log1p(N)


class FloatSlider(QWidget):
    valueChanged = pyqtSignal()

    def __init__(self, min_v: float, max_v: float, decimals: int, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._min, self._max, self._dec = min_v, max_v, decimals
        self._scale = 10 ** decimals
        self._create_ui(name)

    def _create_ui(self, name: str) -> None:
        lab = QLabel(name, self)
        lab.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setRange(0, int((self._max - self._min) * self._scale))
        self._slider.valueChanged.connect(self._refresh)
        self._val = QLabel(self._fmt(self.value()), self)
        self._val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._val.setFixedWidth(80)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(lab, 1)
        lay.addWidget(self._slider, 5)
        lay.addWidget(self._val, 1)

    def _fmt(self, v: float) -> str:
        return f"{v:.{self._dec}f}"

    def _refresh(self, _: int) -> None:
        self._val.setText(self._fmt(self.value()))
        self.valueChanged.emit()

    def value(self) -> float:
        return self._min + self._slider.value() / self._scale

    def set_float_value(self, v: float) -> None:
        v = min(max(v, self._min), self._max)
        self._slider.setValue(int(round((v - self._min) * self._scale)))


class MplCanvas(FigureCanvas):
    def __init__(self, parent: QWidget | None = None):
        fig: Figure = Figure(figsize=(5, 4), dpi=100)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


class ReportWidget(QWidget):
    reportOpened = pyqtSignal(str)

    _SLIDER_BOUNDS: Dict[str, Tuple[float, float]] = {
        "p_inf": (0.0, 1.0),
        "lam": (0.0001, 2.0),
        "gamma": (0.9, 10.0),
        "alpha": (0.0, 0.05),
    }

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.data = None
        self.current_param = "Е50"
        self.current_EGE: Optional[str] = None
        self.param_names = ["Е50", "Eoed", "Eur", "Cu"]
        self.params_store: Dict[Tuple[str, str], Tuple[float, float, float, float]] = {}
        self.results_root: Optional[str] = None  # папка results будет создана после открытия отчёта
        self._build_ui()

    def _build_ui(self):
        self.open_btn = QPushButton("Открыть ведомость"); self.open_btn.clicked.connect(self.open_report)
        self.param_btn = QPushButton(self.current_param)
        self.save_btn = QPushButton("Сохранить")  # новая кнопка сохранения
        self.save_btn.clicked.connect(self.save_current)

        self._build_param_menu()
        top = QHBoxLayout();
        top.addWidget(self.open_btn)
        top.addWidget(self.param_btn)
        top.addWidget(self.save_btn)
        top.addStretch(1)

        self.canvas = MplCanvas(self)
        self.sliders: Dict[str, FloatSlider] = {}
        s_col = QVBoxLayout()
        for name, (mn, mx) in self._SLIDER_BOUNDS.items():
            sl = FloatSlider(mn, mx, 4, name, self); sl.valueChanged.connect(self._on_slider_change)
            self.sliders[name] = sl; s_col.addWidget(sl)
        s_frame = QFrame(); s_frame.setFrameShape(QFrame.Shape.StyledPanel); s_frame.setLayout(s_col)
        self.list_widget = QListWidget(); self.list_widget.setFixedWidth(220); self.list_widget.itemClicked.connect(self._on_ige_clicked)
        left = QVBoxLayout(); left.addWidget(self.canvas); left.addWidget(s_frame)
        content = QHBoxLayout(); content.addLayout(left); content.addWidget(self.list_widget)
        root = QVBoxLayout(self); root.addLayout(top); root.addLayout(content)

    def _build_param_menu(self):
        m = QMenu(self.param_btn)
        for p in self.param_names:
            act = m.addAction(p); act.setCheckable(True); act.setChecked(p == self.current_param); act.triggered.connect(lambda _=False, name=p: self._select_param(name))
        self.param_btn.setMenu(m)

    def _select_param(self, name: str):
        self.current_param = name; self.param_btn.setText(name); self._build_param_menu(); self._load_params(); self._plot()

    def _on_ige_clicked(self, item):
        self.current_EGE = item.text(); self._load_params(); self._plot()

    def _on_slider_change(self):
        self._save_current_params(); self._plot()

    def open_report(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите ведомость", "", "Таблицы Excel (*.xlsx *.xls);;CSV файлы (*.csv);;Все файлы (*)")
        if not path:
            return
        try:
            self.data = load_and_transform(path)

            base_dir = os.path.dirname(path)
            self.results_root = os.path.join(base_dir, "results")
            for folder in ["Е50", "Eoed", "Eur", "Cu"]:
                os.makedirs(os.path.join(self.results_root, folder), exist_ok=True)

            self.list_widget.clear(); self.list_widget.addItems(sorted(self.data["ИГЭ"].dropna().unique()))
            self.current_EGE = None; self.params_store.clear(); self.canvas.axes.clear(); self.canvas.draw_idle(); self.reportOpened.emit(path)
        except Exception:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{traceback.format_exc()}")
            self.data = None

    def save_current(self):
        """Сохранить картинку и параметры в Excel."""
        if self.data is None or self.current_EGE is None or self.results_root is None:
            QMessageBox.warning(self, "Внимание", "Нет данных или не выбрана ИГЭ.")
            return
        try:
            param = self.current_param
            img_dir = os.path.join(self.results_root, param)
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"{param}_{self.current_EGE}.png")
            self.canvas.figure.savefig(img_path, dpi=300)

            p_inf = self.sliders["p_inf"].value()
            lam = self.sliders["lam"].value()
            gamma = self.sliders["gamma"].value()
            alpha = self.sliders["alpha"].value()

            param = self.current_param
            df = self.data[(self.data["ИГЭ"] == self.current_EGE) & (~self.data[param].isna())]
            grp = df.groupby("Цикл промораживания", as_index=False)[param].mean()

            y_min, y_max = grp[param].min(), grp[param].max()

            mean3 = superposition_model(3, p_inf, lam, gamma, alpha) * (
                    y_max - y_min) + y_min
            mean5 = superposition_model(5, p_inf, lam, gamma, alpha) * (
                    y_max - y_min) + y_min

            superposition_model(3, p_inf, lam, gamma, alpha) * (
                    y_max - y_min) + y_min

            summary_path = os.path.join(self.results_root, "summary.xlsx")

            values_3= generate_normal_array(
                mean=mean3,
                sigma=mean3 * 0.2,
                n_points=6,
                accuracy_pct=mean3 / 100
            )

            values_5 = generate_normal_array(
                mean=mean5,
                sigma=mean3 * 0.2,
                n_points=6,
                accuracy_pct=mean5 / 100
            )

            new_row = pd.DataFrame([{
                "ИГЭ": self.current_EGE,
                "p_inf": p_inf,
                "lam": lam,
                "gamma": gamma,
                "alpha": alpha,
                "mean_3": mean3,
                "mean_5": mean5,
                "values_3": ';'.join(f'{x}'.replace('.', ',')  for x in values_3),
                'values_5': ';'.join(f'{x}'.replace('.', ',')  for x in values_5)
            }])

            if os.path.exists(summary_path):
                existing = pd.read_excel(summary_path)
                combined = pd.concat([existing, new_row], ignore_index=True)
            else:
                combined = new_row
            combined.to_excel(summary_path, index=False)

            QMessageBox.information(self, "Готово", f"Сохранено:\n{img_path}\nи обновлён {summary_path}")
        except Exception:
            QMessageBox.critical(self, "Ошибка сохранения", f"Произошла ошибка при сохранении:\n{traceback.format_exc()}")

    def _param_key(self):
        return None if self.current_EGE is None else (self.current_EGE, self.current_param)

    def _save_current_params(self):
        k = self._param_key();
        if k is None:
            return
        self.params_store[k] = (
            self.sliders["p_inf"].value(),
            self.sliders["lam"].value(),
            self.sliders["gamma"].value(),
            self.sliders["alpha"].value(),
        )

    def _load_params(self):
        if self.data is None or self.current_EGE is None:
            return
        k = self._param_key()
        if k not in self.params_store:
            self.params_store[k] = self._fit_initial(k)
        p_inf, lam, gamma, alpha = self.params_store[k]
        self.sliders["p_inf"].set_float_value(p_inf)
        self.sliders["lam"].set_float_value(lam)
        self.sliders["gamma"].set_float_value(gamma)
        self.sliders["alpha"].set_float_value(alpha)

    def _fit_initial(self, k: Tuple[str, str]) -> Tuple[float, float, float, float]:
        ige, param = k
        sub = self.data[(self.data["ИГЭ"] == ige) & (~self.data[param].isna())]
        if sub.empty:
            return 0.5, 0.5, 1.0, 0.1
        grp = sub.groupby("Цикл промораживания", as_index=False)[param].mean()
        N = grp["Цикл промораживания"].to_numpy(float)
        y = grp[param].to_numpy(float)
        y_min, y_max = y.min(), y.max()
        norm_y = np.ones_like(y) if y_max - y_min == 0 else (y - y_min) / (y_max - y_min)
        def _m(n, p_inf, lam, gamma, alpha):
            return superposition_model(n, p_inf, lam, gamma, alpha)
        bounds = ([0.0, 0.0001, 0.1, 0.0], [1.0, 5.0, 5.0, 0.05]); guess = (0.5, 0.5, 1.0, 0.01)
        if curve_fit is not None and len(N) >= 4:
            try:
                popt, _ = curve_fit(_m, N, norm_y, p0=guess, bounds=bounds, maxfev=10000)
            except Exception:
                popt = guess
        else:
            popt = guess
        return tuple(map(float, popt))  # type: ignore


    def _plot(self):
        try:
            self.canvas.axes.clear()
            if self.data is None or self.current_EGE is None:
                self.canvas.draw_idle(); return
            param = self.current_param
            df = self.data[(self.data["ИГЭ"] == self.current_EGE) & (~self.data[param].isna())]
            if df.empty:
                self.canvas.draw_idle(); return
            grp = df.groupby("Цикл промораживания", as_index=False)[param].mean()
            self.canvas.axes.scatter(grp["Цикл промораживания"], grp[param], label="среднее", color="r")
            p_inf = self.sliders["p_inf"].value(); lam = self.sliders["lam"].value(); gamma = self.sliders["gamma"].value(); alpha = self.sliders["alpha"].value()
            N_curve = np.linspace(0, 24, 100)
            y_min, y_max = grp[param].min(), grp[param].max()
            pred = superposition_model(N_curve, p_inf, lam, gamma, alpha) * (y_max - y_min) + y_min
            self.canvas.axes.plot(N_curve, pred, label="модель")

            self.canvas.axes.scatter(np.array([3, 5, 24]), superposition_model(np.array([3, 5, 24]) , p_inf, lam, gamma, alpha) * (y_max - y_min) + y_min, label="предсказания")

            col_labels = ["3", "5"]  # заголовок-строка (2 колонки)
            cell_text = [[f"{np.round(superposition_model(3, p_inf, lam, gamma, alpha) * (y_max - y_min) + y_min, 2)}",  # единственная строка-значений
                          f"{np.round(superposition_model(5, p_inf, lam, gamma, alpha) * (y_max - y_min) + y_min, 2)}"]]

            bbox = [0.6, 0.35, 0.38, 0.3]

            # Вставка таблицы
            table = self.canvas.axes.table(
                cellText=cell_text,
                colLabels=col_labels,
                cellLoc='center',
                bbox=bbox
            )
            table.set_fontsize(14)

            self.canvas.axes.set_xlabel("Цикл промораживания, ед.")
            self.canvas.axes.set_ylabel(f'{param}, МПа')
            self.canvas.axes.set_title(f"ИГЭ {self.current_EGE}")
            self.canvas.axes.grid(True); self.canvas.axes.legend()
            self.canvas.draw_idle()
        except:
            print(traceback.format_exc())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ReportWidget()
    w.resize(1000, 700)
    w.setWindowTitle("Виджет ведомости ИГЭ")
    w.show()
    sys.exit(app.exec())
