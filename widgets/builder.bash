poetry run pyinstaller --noconfirm --clean --onefile --windowed `
    --name ReportWidget `
    --collect-all PyQt6 `
    --collect-submodules matplotlib.backends `
    --hidden-import scipy.optimize `
    widget\main.py