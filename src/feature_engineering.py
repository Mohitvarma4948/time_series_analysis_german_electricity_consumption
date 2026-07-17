from __future__ import annotations

import logging
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from src.config import TABLES_DIR


def build_feature_dataset(logger: logging.Logger | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger('pipeline')
    weekly = pd.read_csv(Path('data/processed/weekly_load_temperature.csv'), parse_dates=['date'])
    weekly = weekly.set_index('date').sort_index()
    weekly = weekly[['load_mw', 'temperature_2m_mean', 'temperature_2m_min', 'temperature_2m_max']].dropna()
    weekly = weekly.copy()
    weekly['week_of_year'] = weekly.index.isocalendar().week.astype(int)
    weekly['month'] = weekly.index.month
    weekly['quarter'] = weekly.index.quarter
    weekly['year'] = weekly.index.year
    weekly['sin_week'] = np.sin(2 * np.pi * weekly['week_of_year'] / 52)
    weekly['cos_week'] = np.cos(2 * np.pi * weekly['week_of_year'] / 52)
    germany_holidays = holidays.country_holidays('DE', years=weekly.index.year.unique())
    weekly['holiday_count'] = 0
    weekly['has_holiday'] = 0
    for idx in weekly.index:
        holidays_for_day = germany_holidays.get(idx.date(), [])
        weekly.loc[idx, 'holiday_count'] = len(holidays_for_day)
        weekly.loc[idx, 'has_holiday'] = int(len(holidays_for_day) > 0)
    for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
        weekly[f'load_lag_{lag}'] = weekly['load_mw'].shift(lag)
    for lag in [1, 2, 4, 8, 52]:
        weekly[f'temp_lag_{lag}'] = weekly['temperature_2m_mean'].shift(lag)
    for window in [4, 8, 13, 26, 52]:
        weekly[f'load_roll_mean_{window}'] = weekly['load_mw'].shift(1).rolling(window).mean()
        weekly[f'load_roll_std_{window}'] = weekly['load_mw'].shift(1).rolling(window).std()
    for window in [4, 8, 13, 26]:
        weekly[f'temp_roll_mean_{window}'] = weekly['temperature_2m_mean'].shift(1).rolling(window).mean()
    weekly = weekly.dropna()
    weekly.reset_index().rename(columns={'date': 'date'}).to_csv(Path('data/processed/weekly_feature_dataset.csv'), index=False)
    feature_dict = pd.DataFrame({'feature': weekly.columns, 'description': ['Weekly load', 'Weekly mean temperature', 'Weekly min temperature', 'Weekly max temperature', 'Week of year', 'Month', 'Quarter', 'Year', 'Sin weekly seasonal', 'Cos weekly seasonal', 'Holiday count', 'Holiday indicator'] + [f'Load lag {lag} weeks' for lag in [1, 2, 3, 4, 8, 12, 26, 52]] + [f'Temperature lag {lag} weeks' for lag in [1, 2, 4, 8, 52]] + [f'Load rolling mean {w} weeks' for w in [4, 8, 13, 26, 52]] + [f'Load rolling std {w} weeks' for w in [4, 8, 13, 26, 52]] + [f'Temperature rolling mean {w} weeks' for w in [4, 8, 13, 26]], 'known_at_forecast_origin': ['known'] * len(weekly.columns)})
    feature_dict.to_csv(TABLES_DIR / 'feature_dictionary.csv', index=False)
    return weekly, feature_dict
