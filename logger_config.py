import sys
import traceback
from loguru import logger

# Удаляем старые хендлеры
logger.remove()

# Включаем backtrace и diagnose, чтобы Loguru сам подхватывал трассировку
logger.add(
    "logs/log.log",
    rotation="1 MB",
    retention="7 days",
    level="INFO",
    backtrace=True,
    diagnose=True
)
logger.add(sys.__stdout__, level="INFO", backtrace=True, diagnose=True)

# Перехват stdout/stderr
class StreamToLogger:
    def __init__(self, level="INFO"):
        self.level = level
        self._buffer = ""
    def write(self, message):
        if message != "\n":
            self._buffer += message
        if "\n" in message:
            self.flush()
    def flush(self):
        if self._buffer:
            logger.log(self.level, self._buffer.rstrip())
            self._buffer = ""

sys.stdout = StreamToLogger("INFO")
sys.stderr = StreamToLogger("ERROR")

# Новый excepthook
def handle_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    # Формируем стек
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    # Логируем на уровне ERROR, включая файл, строку и полную трассировку
    logger.error(f"Uncaught exception:\n{tb_text}")

sys.excepthook = handle_uncaught_exceptions

# Для ошибок в потоках (Python ≥3.8)
import threading
def thread_exception_handler(args):
    tb_text = "".join(traceback.format_exception(
        args.exc_type, args.exc_value, args.exc_traceback
    ))
    logger.error(f"Uncaught thread exception:\n{tb_text}")

threading.excepthook = thread_exception_handler

# Пример
if __name__ == "__main__":
    print("До ошибки")
    raise RuntimeError("Тест необработанного исключения")
