import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from matplotlib.patches import Rectangle

plt.rcParams.update(
    **{
        "font.family": "Times New Roman",
        "font.size": "14",
        "axes.edgecolor": "black",
        "axes.grid": True,
        "figure.subplot.left": 0.1,  # Отступ слева
        "figure.subplot.right": 0.95,  # Отступ справа
        "figure.subplot.bottom": 0.1,  # Отступ снизу
        "figure.subplot.top": 0.9,    # Отступ сверху
        "figure.subplot.wspace": 0.1,  # Расстояние между графиками по горизонтали
        "figure.subplot.hspace": 0.1   # Расстояние между графиками по вертикали
    }
)


def plot_degradation(
        parameters_dict: Dict[str, Dict[str, List[float]]],
        title: str,
        save_path: Optional[str] = None,
        ylabel: str = "Параметр",
        metrics: Optional[Dict] = None,
) -> None:
    """
    Строит график деградации для нескольких наборов данных из словаря.

    :param parameters_dict: словарь с ключами-названиями наборов данных и значениями
                            {'N': [n1, n2, ...], 'D': [d1, d2, ...]}
    :param title: заголовок графика и основа для имени файла при сохранении
    :param save_path: путь к папке для сохранения изображения (если указан)
    :param ylabel: подпись оси Y (по умолчанию "Параметр")
    """
    # Цвета для пунктирных линий (кроме test_data)
    colors = ["blue", "green", "orange", "purple", "brown"]
    fig, ax = plt.subplots(figsize=(10, 6))

    # Базовая сетка
    ax.grid(True, linestyle='--', alpha=0.5)

    # Построение данных
    dash_color_idx = 0
    for key, data in parameters_dict.items():
        N = data.get('N')
        D = data.get('D')

        if key == 'test_data':
            ax.plot(N, D, 'o', color='red', label='test_data', markersize=10)
        elif key == 'additional_test_data':
            ax.plot(N, D, '*', color='red', label='additional_test_data', markersize=12)
        else:
            # Пунктирная линия разного цвета
            color = colors[dash_color_idx % len(colors)]
            ax.plot(N, D, '--', color=color, label=key)
            dash_color_idx += 1

        # Подготовка данных
        cell_text = []
        for key in parameters_dict.keys():
            if key in metrics:
                m = metrics[key]
                cell_text.append([
                    key,
                    f"{m['MAPE']}",
                    f"{m['MAE']}",
                    f"{m['RMSE']}"
                ])
        col_labels = ["Модель", "MAPE", "MAE", "RMSE"]

        bbox = [0.6, 0.35, 0.38, 0.3]

        # Вставка таблицы
        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc='right',
            bbox=bbox
        )
        #table.auto_set_font_size(False)
        table.set_fontsize(14)
        #table.scale(1, 1.2)

        # Подписи осей
    ax.set_xlabel("Количество циклов промораживания n, ед.")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Легенда
    ax.legend()

    # Сохраняем, если указан путь
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{title}.png")
        fig.savefig(file_path, format='png', dpi=300)

    plt.show()


def plot_candlestick(
        stats: pd.DataFrame,
        ige: str,
        save_path: Optional[str] = None,
        ylabel: str = "Параметр",
) -> None:
    """
    Строит график свечей (error bars) для параметра по циклам промораживания для указанного ИГЭ.

    :param stats: DataFrame с колонками ['Цикл промораживания', 'mean', 'min', 'max'],
                  содержащий средние, минимальные и максимальные значения параметра.
    :param ige: идентификатор ИГЭ (например, '1a_t'), используется в заголовке графика.
    :param save_path: путь к папке для сохранения изображения (если указан).
    :param ylabel: подпись оси Y (по умолчанию "Параметр").
    :param metrics: словарь с дополнительными метриками для отображения на графике (не используется в текущей реализации).
    :return: None
    """
    # Вычисление диапазона для свечек
    lower_error = stats['mean'] - stats['min']
    upper_error = stats['max'] - stats['mean']
    asymmetric_error = [lower_error, upper_error]

    # Построение графика
    fig = plt.figure(figsize=(10, 6))
    plt.errorbar(
        stats['Цикл промораживания'],
        stats['mean'],
        yerr=asymmetric_error,
        fmt='-o',
        capsize=5,
        color='blue',
        ecolor='black',
        label='Среднее значение'
    )

    # Настройка графика
    plt.xlabel('Цикл промораживания')
    plt.ylabel(ylabel)
    plt.title(f'Зависимость {ylabel} от циклов промораживания для ИГЭ "{ige}"')
    plt.xticks(stats['Цикл промораживания'])
    plt.grid(True)
    plt.legend()

    # Сохранение графика, если указан путь
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"Зависимость_{ylabel}_ИГЭ_{ige}.png")
        fig.savefig(file_path, format='png', dpi=500, bbox_inches='tight')

    plt.show()


def plot_candlestick_uniform(
    stats: pd.DataFrame,
    ige: str,
    save_path: Optional[str] = None,
    ylabel: str = "Параметр",
    candle_color: str = "tab:blue",  # единый цвет для всех свечей
    mean_line_color: str = "black",  # цвет линии среднего
    body_width: float = 0.6
):
    """
    Рисует кастомные "японские" свечи одного цвета и поверх них линию mean.
    """
    # Подготовка данных
    cycles = stats['Цикл промораживания'].values
    mean = stats['mean'].values
    low  = stats['min'].values
    high = stats['max'].values

    # Тела свечей: mean ± 20% размаха
    delta = (high - low) * 0.2
    open_ = mean - delta
    close = mean + delta

    fig, ax = plt.subplots(figsize=(10,6))

    # Отрисовка каждой свечи одного цвета
    for x, o, c, l, h in zip(cycles, open_, close, low, high):
        # Усик
        ax.vlines(x, l, h, color=candle_color, linewidth=1)

        # Тело свечи
        rect = Rectangle(
            (x - body_width/2, min(o, c)),
            body_width,
            abs(c - o),
            facecolor=candle_color,
            edgecolor='black'
        )
        ax.add_patch(rect)

    # Линия среднего поверх свечей
    ax.plot(
        cycles,
        mean,
        '-',
        linewidth=2,
        color=mean_line_color,
        label='Среднее'
    )

    # Подписи min/max (опционально, можно убрать)
    for x, mn, mx in zip(cycles, low, high):
        ax.text(x, mx + 0.01*(high.max()-low.min()),
                f"{mx:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(x, mn - 0.01*(high.max()-low.min()),
                f"{mn:.2f}", ha="center", va="top",    fontsize=8)

    # Оформление
    ax.set_xlabel("Цикл промораживания n, ед")
    ax.set_ylabel(ylabel)
    ax.set_title(f'Зависимость {ylabel} от циклов промораживания для ИГЭ "{ige}"')
    ax.set_xticks(cycles)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # Сохранение
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fname = f"candlestick_uniform_{ylabel}_{ige}.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')

    plt.show()


def plot_scatter_with_mean(
        ige_df: pd.DataFrame,
        stats: pd.DataFrame,
        ige: str,
        ylabel: str,
        save_path: Optional[str] = None,
) -> None:
    """
    Строит график с индивидуальными значениями параметра и их средними значениями по циклам промораживания для указанного ИГЭ.

    :param ige_df: DataFrame с индивидуальными значениями параметра, содержащий колонки ['Цикл промораживания', ylabel].
    :param stats: DataFrame с колонками ['Цикл промораживания', 'mean'], содержащий средние значения параметра.
    :param ige: идентификатор ИГЭ (например, '1a_t'), используется в заголовке графика.
    :param save_path: путь к папке для сохранения изображения (если указан).
    :param ylabel: подпись оси Y (по умолчанию "Параметр").
    :param metrics: словарь с дополнительными метриками для отображения на графике (не используется в текущей реализации).
    :return: None
    """
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.scatter(ige_df['Цикл промораживания'], ige_df[ylabel], color='blue', label=f'Опытные значения {ylabel}')
    plt.plot(stats['Цикл промораживания'], stats['mean'], 'ro-', label=f'Средние значения {ylabel}')

    # Настройка графика
    plt.xlabel('Цикл промораживания n, ед.')
    plt.ylabel('Е50, МПа')
    plt.title(f'Зависимость Е50 от циклов промораживания для ИГЭ "{ige}"')
    plt.xticks(stats['Цикл промораживания'])
    plt.grid(True)
    plt.legend()

    # Сохранение графика, если указан путь
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"Зависимость_{ylabel}_ИГЭ_{ige}.png")
        plt.savefig(file_path, format='png', dpi=300, bbox_inches='tight')

    plt.show()


# Пример использования
if __name__ == "__main__":
    parameters_dict = {
        'test_data': {
            'N': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'D': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        'arina_data': {
            'N': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'D': [2, 2, 4, 6, 8, 10, 12, 14, 16, 18],
        },
        'LSM_data': {
            'N': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'D': [1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        },
        'holt_data': {
            'N': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'D': [0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9.0],
        },
    }
    plot_degradation(parameters_dict, "Пример_графика", save_path=None, ylabel="Относительная деградация")
