#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Виджет ведомости ИГЭ — candlestick + sliders + min/max annotation (2025-06-17)
"""
import sys, os, traceback
from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QMenu,
    QMessageBox, QPushButton, QVBoxLayout, QWidget, QDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QSlider
)
import warnings
warnings.filterwarnings("ignore")

from scipy.optimize import curve_fit      # type: ignore


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

# ------------------------------------------------------------------------------
# Утилиты
# ------------------------------------------------------------------------------

def generate_normal_array(
    mean: Union[int, float],
    sigma: Union[int, float],
    n_points: int,
    accuracy_pct: Union[int, float] = 0.5
) -> np.ndarray:
    if n_points <= 0:
        raise ValueError("n_points должен быть положительным целым числом")
    if sigma < 0:
        raise ValueError("sigma должно быть неотрицательным")
    if accuracy_pct < 0:
        raise ValueError("accuracy_pct не может быть отрицательным")

    rng = np.random.default_rng()
    tolerance = abs(mean) * (accuracy_pct / 100.0)

    for _ in range(10):
        sample = rng.normal(loc=mean, scale=sigma, size=n_points)
        sample += mean - sample.mean()
        if abs(sample.mean() - mean) <= tolerance:
            return sample
    return sample - (sample.mean() - mean)


def load_and_transform(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name='Лист1', header=0)
    properties = ['Е50', 'Eoed', 'Eur', 'Cu']
    id_columns = ['№', 'ИГЭ']
    cycles = [0, 3, 5, 10]
    suffixes = ['-0', '-3', '-5', '-10']

    dfs = []
    for cycle, suf in zip(cycles, suffixes):
        cols = id_columns + [p + suf for p in properties]
        tmp = df[cols].copy()
        tmp.columns = id_columns + properties
        tmp['Цикл промораживания'] = cycle
        dfs.append(tmp)

    long_df = (pd.concat(dfs, ignore_index=True)
                 .sort_values(['№', 'Цикл промораживания'])
                 .reset_index(drop=True))
    return long_df


def superposition_model(N: np.ndarray, p_inf: float, lam: float, gamma: float, alpha: float) -> np.ndarray:
    """p(N) = p_inf + (1−p_inf)·exp[−(λN)^γ] − α·ln(1+N)."""
    return p_inf + (1.0 - p_inf) * np.exp(-(lam * N) ** gamma) - alpha * np.log1p(N)

# ------------------------------------------------------------------------------
# GUI-элементы
# ------------------------------------------------------------------------------

class FloatSlider(QWidget):
    valueChanged = pyqtSignal()

    def __init__(self, min_v: float, max_v: float, decimals: int, name: str, parent=None):
        super().__init__(parent)
        self._min, self._max, self._dec = min_v, max_v, decimals
        self._scale = 10 ** decimals
        self._create_ui(name)

    def _create_ui(self, name: str):
        lab = QLabel(name, self)
        lab.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setRange(0, int((self._max - self._min) * self._scale))
        self._slider.valueChanged.connect(self._refresh)
        self._val = QLabel(self._fmt(self.value()), self)
        self._val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._val.setFixedWidth(80)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(lab, 1); lay.addWidget(self._slider, 5); lay.addWidget(self._val, 1)

    def _fmt(self, v: float) -> str:
        return f"{v:.{self._dec}f}"

    def _refresh(self, _: int):
        self._val.setText(self._fmt(self.value()))
        self.valueChanged.emit()

    def value(self) -> float:
        return self._min + self._slider.value() / self._scale

    def set_float_value(self, v: float):
        v = min(max(v, self._min), self._max)
        self._slider.setValue(int(round((v - self._min) * self._scale)))


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig: Figure = Figure(figsize=(5, 4), dpi=100)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


class ParamsDialog(QDialog):
    """Диалог с текущими параметрами модели."""
    def __init__(self, params: Tuple[float, float, float, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Параметры модели")
        labels = ["p_inf", "λ", "γ", "α"]
        tbl = QTableWidget(1, 4, self)
        tbl.setHorizontalHeaderLabels(labels)
        for i, v in enumerate(params):
            tbl.setItem(0, i, QTableWidgetItem(f"{v:.6f}"))
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout = QVBoxLayout(self); layout.addWidget(tbl)

# ------------------------------------------------------------------------------
# Основной виджет
# ------------------------------------------------------------------------------

class ReportWidget(QWidget):
    reportOpened = pyqtSignal(str)

    _SLIDER_BOUNDS: Dict[str, Tuple[float, float]] = {
        "p_inf": (0.0, 1.0),
        "lam": (0.0001, 2.0),
        "gamma": (0.9, 10.0),
        "alpha": (0.0, 0.05),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data: Optional[pd.DataFrame] = None
        self.current_param = "Е50"
        self.current_EGE: Optional[str] = None
        self.param_names = ["Е50", "Eoed", "Eur", "Cu"]
        self.params_store: Dict[Tuple[str, str], Tuple[float, float, float, float]] = {}
        self.results_root: Optional[str] = None
        self._build_ui()

    def _build_ui(self):
        self.open_btn = QPushButton("Открыть ведомость"); self.open_btn.clicked.connect(self.open_report)
        self.param_btn = QPushButton(self.current_param)
        self.save_btn = QPushButton("Сохранить"); self.save_btn.clicked.connect(self.save_current)
        self.params_btn = QPushButton("Параметры"); self.params_btn.clicked.connect(self._show_params_dialog)

        self._build_param_menu()
        top = QHBoxLayout()
        top.addWidget(self.open_btn); top.addWidget(self.param_btn)
        top.addWidget(self.params_btn); top.addWidget(self.save_btn); top.addStretch(1)

        self.canvas = MplCanvas(self)

        # --------- слайдеры ----------
        self.sliders: Dict[str, FloatSlider] = {}
        s_col = QVBoxLayout()
        for name, (mn, mx) in self._SLIDER_BOUNDS.items():
            sl = FloatSlider(mn, mx, 4, name, self)
            sl.valueChanged.connect(self._on_slider_change)
            self.sliders[name] = sl
            s_col.addWidget(sl)
        s_frame = QFrame(); s_frame.setFrameShape(QFrame.Shape.StyledPanel)
        s_frame.setLayout(s_col)

        # --------- список ИГЭ --------
        self.list_widget = QListWidget(); self.list_widget.setFixedWidth(220)
        self.list_widget.itemClicked.connect(self._on_ige_clicked)

        # --------- компоновка --------
        left = QVBoxLayout(); left.addWidget(self.canvas); left.addWidget(s_frame)
        content = QHBoxLayout(); content.addLayout(left); content.addWidget(self.list_widget)
        root = QVBoxLayout(self); root.addLayout(top); root.addLayout(content)

    def _build_param_menu(self):
        m = QMenu(self.param_btn)
        for p in self.param_names:
            act = m.addAction(p); act.setCheckable(True)
            act.setChecked(p == self.current_param)
            act.triggered.connect(lambda _=False, name=p: self._select_param(name))
        self.param_btn.setMenu(m)

    def _select_param(self, name: str):
        self.current_param = name
        self.param_btn.setText(name)
        self._build_param_menu()
        self._fit_and_cache_params()
        self._load_params_to_sliders()
        self._plot()

    def _on_ige_clicked(self, item):
        self.current_EGE = item.text()
        self._fit_and_cache_params()
        self._load_params_to_sliders()
        self._plot()

    def _on_slider_change(self):
        self._save_current_params()
        self._plot()

    def _show_params_dialog(self):
        k = self._param_key()
        if k is None or k not in self.params_store:
            QMessageBox.information(self, "Информация", "Параметры не рассчитаны.")
            return
        dlg = ParamsDialog(self.params_store[k], self)
        dlg.exec()

    def open_report(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите ведомость", "",
            "Таблицы Excel (*.xlsx *.xls);;CSV файлы (*.csv);;Все файлы (*)"
        )
        if not path:
            return
        try:
            self.data = load_and_transform(path)
            base_dir = os.path.dirname(path)
            self.results_root = os.path.join(base_dir, "results")
            for folder in self.param_names:
                os.makedirs(os.path.join(self.results_root, folder), exist_ok=True)

            self.list_widget.clear()
            self.list_widget.addItems(sorted(self.data["ИГЭ"].dropna().unique()))
            self.current_EGE = None
            self.params_store.clear()
            self.canvas.axes.clear(); self.canvas.draw_idle()
            self.reportOpened.emit(path)
        except Exception:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{traceback.format_exc()}")
            self.data = None

    def _param_key(self):
        return None if self.current_EGE is None else (self.current_EGE, self.current_param)

    def _fit_and_cache_params(self):
        if self.data is None or self.current_EGE is None:
            return
        k = self._param_key()
        if k in self.params_store:
            return

        sub = self.data[(self.data["ИГЭ"] == self.current_EGE) & (~self.data[self.current_param].isna())]
        if sub.empty:
            self.params_store[k] = (0.5, 0.5, 1.0, 0.01)
            return

        grp = (sub.groupby("Цикл промораживания", as_index=False)[self.current_param]
                  .mean())
        grp = grp.set_index("Цикл промораживания").reindex([0, 3, 5, 10]).dropna().reset_index()
        N = grp["Цикл промораживания"].to_numpy(float)
        y = grp[self.current_param].to_numpy(float)

        y_min, y_max = y.min(), y.max()
        norm_y = np.ones_like(y) if y_max == y_min else (y - y_min) / (y_max - y_min)

        def _m(n, p_inf, lam, gamma, alpha):
            return superposition_model(n, p_inf, lam, gamma, alpha)

        bounds = ([0.0, 0.0001, 0.1, 0.0], [1.0, 5.0, 10.0, 0.05])
        guess = (0.6, 0.5, 2.0, 0.01)
        if curve_fit is not None and len(N) >= 4:
            try:
                popt, _ = curve_fit(_m, N, norm_y, p0=guess, bounds=bounds, maxfev=10000)
            except Exception:
                popt = guess
        else:
            popt = guess
        self.params_store[k] = tuple(map(float, popt))

    def _load_params_to_sliders(self):
        k = self._param_key()
        if k is None or k not in self.params_store:
            return
        p_inf, lam, gamma, alpha = self.params_store[k]
        self.sliders["p_inf"].set_float_value(p_inf)
        self.sliders["lam"].set_float_value(lam)
        self.sliders["gamma"].set_float_value(gamma)
        self.sliders["alpha"].set_float_value(alpha)

    def _save_current_params(self):
        k = self._param_key()
        if k is None:
            return
        self.params_store[k] = (
            self.sliders["p_inf"].value(),
            self.sliders["lam"].value(),
            self.sliders["gamma"].value(),
            self.sliders["alpha"].value(),
        )

    def save_current(self):
        """
        Сохраняет:
          • параметры модели;
          • реальные средние по 0/3/5/10 циклам (как на свечах);
          • прогноз модели на 24-й цикл.

        Результат пишет в summary.xlsx и сохраняет PNG-график.
        """
        if self.data is None or self.current_EGE is None or self.results_root is None:
            QMessageBox.warning(self, "Внимание", "Нет данных или не выбрана ИГЭ.")
            return

        try:
            param = self.current_param
            img_dir = os.path.join(self.results_root, param)
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"{param}_{self.current_EGE}.png")
            self.canvas.figure.savefig(img_path, dpi=300)

            # ---------- параметры модели -------------
            p_inf  = self.sliders["p_inf"].value()
            lam    = self.sliders["lam"].value()
            gamma  = self.sliders["gamma"].value()
            alpha  = self.sliders["alpha"].value()

            # ---------- расчёт средних -------------
            df  = self.data[(self.data["ИГЭ"] == self.current_EGE) & (~self.data[param].isna())]
            grp = df.groupby("Цикл промораживания", as_index=False)[param].mean()

            # реальные средние
            means = grp.set_index("Цикл промораживания")[param].reindex([0, 3, 5, 10])
            mean0  = float(means.loc[0])  if not pd.isna(means.loc[0])  else np.nan
            mean3  = float(means.loc[3])  if not pd.isna(means.loc[3])  else np.nan
            mean5  = float(means.loc[5])  if not pd.isna(means.loc[5])  else np.nan
            mean10 = float(means.loc[10]) if not pd.isna(means.loc[10]) else np.nan

            # диапазон для нормирования, как в _plot()
            y_min, y_max = grp[param].min(), grp[param].max()

            # прогноз для 24 циклов
            pred24 = superposition_model(24, p_inf, lam, gamma, alpha) * (y_max - y_min) + y_min

            summary_path = os.path.join(self.results_root, "summary.xlsx")
            new_row = pd.DataFrame([{
                "ИГЭ":     self.current_EGE,
                "p_inf":   p_inf,
                "lam":     lam,
                "gamma":   gamma,
                "alpha":   alpha,
                "mean_0":  mean0,
                "mean_3":  mean3,
                "mean_5":  mean5,
                "mean_10": mean10,
                "pred_24": pred24
            }])

            if os.path.exists(summary_path):
                combined = pd.concat([pd.read_excel(summary_path), new_row], ignore_index=True)
            else:
                combined = new_row

            combined.to_excel(summary_path, index=False)

            QMessageBox.information(
                self,
                "Готово",
                f"Сохранено PNG:\n{img_path}\nи обновлён файл {summary_path}"
            )

        except Exception:
            QMessageBox.critical(
                self,
                "Ошибка сохранения",
                f"Произошла ошибка при сохранении:\n{traceback.format_exc()}"
            )

    def _plot(self) -> None:
        try:
            ax = self.canvas.axes
            ax.clear()

            # ------------------------------------------------------------
            # 1. Подготовка данных
            # ------------------------------------------------------------
            if self.data is None or self.current_EGE is None:
                self.canvas.draw_idle()
                return

            param = self.current_param
            df = self.data[
                (self.data["ИГЭ"] == self.current_EGE) & (~self.data[param].isna())
                ]
            if df.empty:
                self.canvas.draw_idle()
                return

            cycles = [0, 3, 5, 10]
            data_by_cycle = [
                df[df["Цикл промораживания"] == n][param].to_numpy(float) for n in cycles
            ]

            # ------------------------------------------------------------
            # 2. Box-plot без выбросов
            # ------------------------------------------------------------
            bp = ax.boxplot(
                data_by_cycle,
                positions=cycles,
                widths=0.8,
                showmeans=True,
                meanline=True,
                showfliers=False,
                patch_artist=True,
                meanprops=dict(color="navy", lw=1.5, ls="--"),
                medianprops=dict(color="black", lw=1.2),
            )
            for box in bp["boxes"]:
                box.set(facecolor="royalblue", alpha=0.55, edgecolor="black")

            # ------------------------------------------------------------
            # 3. Горизонтальные подписи min / mean / max
            # ------------------------------------------------------------

            # ---------- 3. Подписи whisker-min / mean / whisker-max ----------
            for x, vals in zip(cycles, data_by_cycle):
                if vals.size == 0:
                    continue

                q1, q3 = np.percentile(vals, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                lower_whisk = vals[vals >= lower_bound].min()  # FIX ➋
                upper_whisk = vals[vals <= upper_bound].max()  # FIX ➋
                vmean = vals.mean()

                # верх
                ax.text(
                    x, upper_whisk + 0.03 * (upper_whisk - lower_whisk),
                    f"{upper_whisk:.1f}" if upper_whisk > 1 else f"{upper_whisk:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )
                # низ
                ax.text(
                    x, lower_whisk - 0.03 * (upper_whisk - lower_whisk),
                    f"{lower_whisk:.1f}" if lower_whisk > 1 else f"{lower_whisk:.3f}",
                    ha="center", va="top", fontsize=7,
                )
            # ------------------------------------------------------------
            # 4. Кривая модели и «середины» свечей
            # ------------------------------------------------------------
            p_inf = self.sliders["p_inf"].value()
            lam = self.sliders["lam"].value()
            gamma = self.sliders["gamma"].value()
            alpha = self.sliders["alpha"].value()

            # NEW: безопасный расчёт средних без ворнингов
            means = np.array(
                [v.mean() if v.size else np.nan for v in data_by_cycle], dtype=float
            )
            not_nan = ~np.isnan(means)
            if not not_nan.any():  # на всякий случай
                self.canvas.draw_idle()
                return

            y0, y1 = np.nanmin(means), np.nanmax(means)

            N = np.linspace(0, 24, 240)
            pred = superposition_model(N, p_inf, lam, gamma, alpha) * (y1 - y0) + y0
            ax.plot(N, pred, lw=2, color="black", ls="--", label="model")

            y24 = superposition_model(24, p_inf, lam, gamma, alpha) * (y1 - y0) + y0

            centers_x = list(np.array(cycles)[not_nan])
            centers_y = list(means[not_nan])
            ax.plot(
                centers_x,
                centers_y,
                "-o",
                color="darkorange",
                mfc="white",
                ms=6,
                label="mean values",
            )
            ax.plot(24, y24, "s", ms=8, color="red", zorder=3, label="prediction 24")

            # ------------------------------------------------------------
            # 5. Оформление осей
            # ------------------------------------------------------------
            all_vals = np.concatenate([v for v in data_by_cycle if v.size])
            pad = 0.10 * np.ptp(all_vals) if all_vals.size else 1.0
            ax.set_ylim(all_vals.min() - pad, all_vals.max() + pad)
            ax.set_xlim(-1.5, 26)
            xticks = sorted(cycles + [24])
            ax.set_xticks(xticks)
            ax.set_xticklabels(map(str, xticks))
            ax.set_xlabel("Цикл промораживания, ед.")
            ax.set_ylabel(f"{param}, МПа")
            ax.set_title(f"ИГЭ {self.current_EGE}")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend()

            self.canvas.draw_idle()

        except Exception:
            print(traceback.format_exc())


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ReportWidget()
    w.resize(1100, 780)
    w.setWindowTitle("Виджет ведомости ИГЭ — candlestick + sliders + min/max")
    w.show()
    sys.exit(app.exec())
