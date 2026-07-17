from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import holidays
import numpy as np
import pandas as pd

from src.config import (
    DAILY_OUTPUT,
    ELECTRICITY_FILE,
    LOAD_COLUMN,
    LOAD_OUTPUT,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    TEMPERATURE_FILE,
    TIMESTAMP_COLUMN,
    WEEKLY_MERGED_OUTPUT,
    WEEKLY_OUTPUT,
    WEEKLY_TEMP_OUTPUT,
)
from src.utils import as_datetime, ensure_columns, ensure_directory


def read_temperature_data(path: Path = TEMPERATURE_FILE) -> pd.DataFrame:
    header_row = None
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            first_cell = line.split(',', 1)[0].strip().lower()
            if first_cell in {'time', 'date'}:
                header_row = i
                break
    if header_row is None:
        raise ValueError(f'Could not locate the daily temperature header row in {path}')

    temp_df = pd.read_csv(path, skiprows=header_row)
    temp_df = temp_df.rename(columns={
        'time': 'date',
        'temperature_2m_mean (°C)': 'temperature_2m_mean',
        'temperature_2m_max (°C)': 'temperature_2m_max',
        'temperature_2m_min (°C)': 'temperature_2m_min',
    })
    ensure_columns(
        temp_df,
        ['date', 'temperature_2m_mean', 'temperature_2m_min', 'temperature_2m_max'],
        'temperature data',
    )
    temp_df['date'] = pd.to_datetime(temp_df['date'], utc=True, errors='coerce')
    temp_df = temp_df.dropna(subset=['date']).sort_values('date').set_index('date')
    return temp_df[['temperature_2m_mean', 'temperature_2m_min', 'temperature_2m_max']].apply(pd.to_numeric, errors='coerce')


def inspect_raw_data() -> dict[str, Any]:
    electricity = pd.read_csv(ELECTRICITY_FILE)
    temperature_raw = pd.read_csv(TEMPERATURE_FILE)
    temperature_daily = read_temperature_data(TEMPERATURE_FILE)
    summary = {
        'electricity_shape': electricity.shape,
        'electricity_columns': electricity.columns.tolist(),
        'temperature_shape': temperature_raw.shape,
        'temperature_columns': temperature_raw.columns.tolist(),
        'temperature_daily_shape': temperature_daily.shape,
        'temperature_daily_columns': temperature_daily.columns.tolist(),
        'electricity_dtypes': {k: str(v) for k, v in electricity.dtypes.items()},
        'temperature_dtypes': {k: str(v) for k, v in temperature_daily.dtypes.items()},
    }
    print('Raw inspection summary:')
    print(f"Electricity shape: {summary['electricity_shape']}")
    print(f"Electricity columns: {summary['electricity_columns'][:10]}...")
    print(f"Temperature shape: {summary['temperature_shape']}")
    print(f"Temperature columns: {summary['temperature_columns']}")
    print(f"Parsed daily temperature shape: {summary['temperature_daily_shape']}")
    print(f"Parsed daily temperature columns: {summary['temperature_daily_columns']}")
    return summary


def prepare_data(logger: logging.Logger | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger('pipeline')
    electricity = pd.read_csv(ELECTRICITY_FILE)
    temperature_raw = read_temperature_data(TEMPERATURE_FILE)
    ensure_columns(electricity, [TIMESTAMP_COLUMN, LOAD_COLUMN], 'electricity data')

    electricity[TIMESTAMP_COLUMN] = as_datetime(electricity[TIMESTAMP_COLUMN])
    electricity = electricity.dropna(subset=[TIMESTAMP_COLUMN]).copy()
    electricity = electricity.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    electricity = electricity.loc[electricity[TIMESTAMP_COLUMN] >= '2015-01-01'].copy()
    electricity = electricity.rename(columns={LOAD_COLUMN: 'load_mw'})
    electricity = electricity[['utc_timestamp', 'load_mw']].copy()

    electricity = electricity.set_index('utc_timestamp').sort_index()
    electricity = electricity.asfreq('h')
    electricity['load_mw'] = electricity['load_mw'].astype(float)

    short_gap_mask = electricity['load_mw'].isna()
    if short_gap_mask.any():
        gap_sizes = electricity['load_mw'].isna().astype(int).groupby((electricity['load_mw'].isna().astype(int).diff().fillna(0) != 0).cumsum()).sum()
        long_gap_mask = gap_sizes > 3
        to_interpolate = electricity.index[short_gap_mask & ~gap_sizes.reindex(electricity.index).fillna(0).astype(int).gt(3)]
    else:
        to_interpolate = []
    if len(to_interpolate) > 0:
        electricity.loc[to_interpolate, 'load_mw'] = electricity['load_mw'].interpolate(method='time')

    quality_rows = []
    quality_rows.append({'step': 'load_column_selection', 'details': f'Used {LOAD_COLUMN} as German load series'})
    quality_rows.append({'step': 'date_filter', 'details': 'Retained timestamps from 2015-01-01 onward'})
    quality_rows.append({'step': 'timestamp_sort', 'details': 'Sorted by utc_timestamp'})
    quality_rows.append({'step': 'resample', 'details': 'Reindexed to hourly frequency'})
    quality_rows.append({'step': 'gap_handling', 'details': 'Interpolated only short gaps; long gaps left as NaN'})

    # Temperature preparation
    temp_df = temperature_raw.copy()

    # Weekly aggregation
    daily_load = electricity.resample('D').mean().dropna()
    weekly_load = electricity.resample('W-SUN').mean().dropna()
    weekly_temp = temp_df.resample('D').mean().dropna().resample('W-SUN').mean()

    weekly_df = weekly_load.join(weekly_temp, how='left')
    weekly_df = weekly_df.reset_index().rename(columns={'utc_timestamp': 'date'})
    weekly_df['date'] = pd.to_datetime(weekly_df['date'], utc=True)
    weekly_df = weekly_df.set_index('date').sort_index()
    weekly_load = weekly_load.reset_index().rename(columns={'utc_timestamp': 'date'})
    weekly_load['date'] = pd.to_datetime(weekly_load['date'], utc=True)
    weekly_temp = weekly_temp.reset_index().rename(columns={'utc_timestamp': 'date'})
    weekly_temp['date'] = pd.to_datetime(weekly_temp['date'], utc=True)
    weekly_df['holiday_count'] = 0
    weekly_df['has_holiday'] = 0
    germany_holidays = holidays.country_holidays('DE', years=weekly_df.index.year.unique())
    for idx in weekly_df.index:
        date = idx.to_pydatetime().date()
        hols = germany_holidays.get(date, [])
        weekly_df.loc[idx, 'holiday_count'] = len(hols)
        weekly_df.loc[idx, 'has_holiday'] = int(len(hols) > 0)

    weekly_df.reset_index().to_csv(WEEKLY_MERGED_OUTPUT, index=False)
    weekly_load.to_csv(WEEKLY_OUTPUT, index=False)
    weekly_temp.to_csv(WEEKLY_TEMP_OUTPUT, index=False)
    daily_load.to_csv(DAILY_OUTPUT)
    electricity.reset_index().to_csv(LOAD_OUTPUT, index=False)

    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(PROCESSED_DATA_DIR / 'data_quality_summary.csv', index=False)
    quality_df.to_csv(TABLES_DIR / 'data_quality_summary.csv', index=False)

    return electricity, daily_load, weekly_load, weekly_temp, weekly_df, quality_df
