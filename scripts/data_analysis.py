import pandas as pd
from typing import Tuple
from pathlib import Path

from utils.plotters import plot_candlestick, plot_scatter_with_mean, plot_candlestick_uniform


def sort_params_for_ige(long_df: pd.DataFrame, ige: str, param:str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ige_df = long_df[long_df['№ ИГЭ/РГЭ'] == ige]

    # Проверка наличия данных
    if ige_df.empty:
        raise ValueError(f"Нет данных для ИГЭ '{ige}'.")
    else:
        # Группировка и расчет статистики
        stats = ige_df.groupby('Цикл промораживания')[param].agg(['mean', 'min', 'max']).reset_index()
        stats = stats.sort_values('Цикл промораживания')

        if stats.empty:
            raise ValueError(f"Нет данных {param} для ИГЭ '{ige}'.")
        else:
            return stats, ige_df



if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    data_file = root / "data" / "data.csv"
    res_dir = root / "results"

    long_df = pd.read_csv(data_file)

    ige = '58_3а'
    param = 'Е50'

    stats, ige_df = sort_params_for_ige(
        long_df, ige, param
    )


    plot_scatter_with_mean(
        ige_df=ige_df,
        stats=stats,
        ige=ige,
        ylabel=param,
        save_path=res_dir
    )

    plot_candlestick_uniform(
        stats=stats,
        ige=ige,
        ylabel=param,
        save_path=res_dir
    )