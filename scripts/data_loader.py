import pandas as pd
from pathlib import Path


def load_and_transform(file_path: str) -> pd.DataFrame:
    """
    Загружает Excel-файл с двумя строками заголовков,
    преобразует в таблицу с колонками:
    sample_id, freezing_cycles и все параметры в отдельных столбцах.
    """
    # Получаем готовый DataFrame
    df = pd.read_excel(
        file_path,
        sheet_name='Переделка Лист 1',
        header=0
    )

    # Список свойств
    properties = ['rs', 'r', 'rd', 'n', 'e', 'W', 'Sr', 'WL', 'WP', 'Ip', 'IL', 'Ir', 'Е50', 'E', 'Мю', 'Cu']

    # Уникальные столбцы
    id_columns = ['Лаб. № пробы', 'Наименование выработки', 'Глубина отбора образца, м', 'Наименование грунта',
                  '№ ИГЭ/РГЭ']

    # Циклы и соответствующие суффиксы
    cycles = [0, 3, 5, 10, 24]
    suffixes = ['', '.1', '.2', '.3', '.4']

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

    # Добавляем 'Kd' и 'Cud', если они есть в исходном DataFrame
    if 'Kd' in df.columns and 'Cud' in df.columns:
        long_df = long_df.merge(df[['Лаб. № пробы', 'Kd', 'Cud']], on='Лаб. № пробы', how='left')

    # Сортируем по 'Лаб. № пробы' и 'Цикл промораживания'
    long_df = long_df.sort_values(by=['Лаб. № пробы', 'Цикл промораживания']).reset_index(drop=True)

    return long_df


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    excel_file = root / "data" / "Сводная ведомость.xlsx"
    data_file = root / "data" / "data.csv"
    long_df = load_and_transform(excel_file)
    long_df.to_csv(data_file, index=False)
