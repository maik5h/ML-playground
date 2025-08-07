from pathlib import Path

from src.config import load_config
from src.regression_playground import start_regression


if __name__ == "__main__":
    base = Path().resolve()

    load_config(base / 'config.json')
    start_regression()
