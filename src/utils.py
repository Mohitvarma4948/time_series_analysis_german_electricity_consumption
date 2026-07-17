from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import LOGS_DIR, SEED


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def setup_logger(name: str = 'pipeline', log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if log_file:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_columns(df: pd.DataFrame, required: list[str], context: str = 'dataframe') -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'{context} is missing required columns: {missing}')


def as_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors='coerce')


def calculate_metrics(actual: pd.Series, forecast: pd.Series, seasonal_period: int = 52) -> pd.DataFrame:
    actual = pd.Series(actual).astype(float)
    forecast = pd.Series(forecast).astype(float)
    error = actual - forecast
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    mape = np.mean(np.abs(error / actual)) * 100
    smape = np.mean(2 * np.abs(error) / (np.abs(actual) + np.abs(forecast))) * 100
    naive_errors = np.abs(actual.diff(seasonal_period).dropna())
    if len(naive_errors) == 0:
        mase = np.nan
    else:
        mase = np.mean(np.abs(error.dropna())) / np.mean(naive_errors)
    return pd.DataFrame({
        'RMSE': [rmse],
        'MAE': [mae],
        'MAPE': [mape],
        'sMAPE': [smape],
        'MASE': [mase],
    })
