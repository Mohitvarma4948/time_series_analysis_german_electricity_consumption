from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, TABLES_DIR
from src.utils import calculate_metrics


def make_split(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = weekly.index.max() - pd.DateOffset(years=2)
    train = weekly[weekly.index <= train_end]
    test = weekly[weekly.index > train_end]
    return train, test


def historical_mean_forecast(train: pd.Series, horizon: int) -> pd.Series:
    return pd.Series([train.mean()] * horizon, index=range(horizon))


def naive_forecast(train: pd.Series, horizon: int) -> pd.Series:
    return pd.Series([train.iloc[-1]] * horizon, index=range(horizon))


def seasonal_naive_forecast(train: pd.Series, horizon: int, season: int = 52) -> pd.Series:
    last_values = train.iloc[-season:]
    return pd.Series([last_values.iloc[i % season] for i in range(horizon)], index=range(horizon))


def drift_forecast(train: pd.Series, horizon: int) -> pd.Series:
    last = train.iloc[-1]
    first = train.iloc[0]
    slope = (last - first) / (len(train) - 1)
    return pd.Series([last + slope * (i + 1) for i in range(horizon)], index=range(horizon))


def evaluate_benchmarks(logger: logging.Logger | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger('pipeline')
    weekly = pd.read_csv(Path('data/processed/weekly_load_temperature.csv'), parse_dates=['date'])
    weekly = weekly.set_index('date').sort_index()
    weekly = weekly[['load_mw']].dropna()
    train, test = make_split(weekly)
    actual = test['load_mw'].reset_index(drop=True)
    models = {
        'Historical Mean': historical_mean_forecast(train['load_mw'], len(test)),
        'Naive': naive_forecast(train['load_mw'], len(test)),
        'Seasonal Naive': seasonal_naive_forecast(train['load_mw'], len(test), season=52),
        'Drift': drift_forecast(train['load_mw'], len(test)),
    }
    rows = []
    for name, forecast in models.items():
        metrics = calculate_metrics(actual, forecast, seasonal_period=52)
        metrics = metrics.reset_index(drop=True)
        metrics['Model'] = name
        rows.append(metrics)
    metrics_df = pd.concat(rows, ignore_index=True)
    metrics_df.to_csv(TABLES_DIR / 'benchmark_metrics.csv', index=False)
    forecasts_df = pd.DataFrame({'actual': actual, **{k: v.reset_index(drop=True) for k, v in models.items()}})
    forecasts_df.to_csv(TABLES_DIR / 'benchmark_forecasts.csv', index=False)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    dates = test.index
    ax.plot(dates, forecasts_df['actual'], color='black', linewidth=2, label='Actual')
    for name in models:
        ax.plot(dates, forecasts_df[name], linewidth=1.2, label=name)
    ax.set_title('Benchmark weekly forecasts')
    ax.set_ylabel('Average load (MW)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'benchmark_forecasts.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return metrics_df, forecasts_df
