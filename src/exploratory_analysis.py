from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL

from src.config import DAILY_OUTPUT, FIGURES_DIR, LOAD_OUTPUT, WEEKLY_MERGED_OUTPUT, WEEKLY_OUTPUT


def _read_indexed_series(path: Path, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df = df.set_index(date_col).sort_index()
    df.index = pd.DatetimeIndex(df.index)
    df['load_mw'] = pd.to_numeric(df['load_mw'], errors='coerce')
    return df


def plot_series(logger: logging.Logger | None = None) -> None:
    if logger is None:
        logger = logging.getLogger('pipeline')
    hourly = _read_indexed_series(LOAD_OUTPUT, 'utc_timestamp')
    daily = _read_indexed_series(DAILY_OUTPUT, 'utc_timestamp')
    weekly_load = _read_indexed_series(WEEKLY_OUTPUT, 'date')
    weekly = pd.read_csv(WEEKLY_MERGED_OUTPUT, parse_dates=['date'])
    weekly['date'] = pd.to_datetime(weekly['date'], utc=True)
    weekly = weekly.set_index('date').sort_index()
    weekly.index = pd.DatetimeIndex(weekly.index)
    weekly['load_mw'] = pd.to_numeric(weekly['load_mw'], errors='coerce')

    weekday_profile = hourly.groupby(hourly.index.dayofweek)['load_mw'].mean().reindex(range(7))
    hour_profile = hourly.groupby(hourly.index.hour)['load_mw'].mean().reindex(range(24))
    weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    fig, axes = plt.subplots(3, 3, figsize=(18, 15), dpi=200)
    axes = axes.flatten()
    hourly_display = hourly['load_mw'].resample('D').mean()
    hourly_display.plot(ax=axes[0], title='Hourly electricity demand series (daily display average)')
    daily['load_mw'].plot(ax=axes[1], title='Daily electricity demand')
    weekly_load['load_mw'].plot(ax=axes[2], title='Weekly electricity demand')
    weekly_load['load_mw'].tail(104).plot(ax=axes[3], title='Last two years of weekly demand')
    weekly_load.groupby(weekly_load.index.month)['load_mw'].mean().reindex(range(1, 13)).plot(
        ax=axes[4], marker='o', title='Monthly seasonal profile'
    )
    axes[5].plot(range(7), weekday_profile.values, marker='o')
    axes[5].set_xticks(range(7), weekday_labels, rotation=35, ha='right')
    axes[5].set_title('Day-of-week profile')
    axes[6].plot(range(24), hour_profile.values, marker='o')
    axes[6].set_xticks(range(24))
    axes[6].set_title('Hour-of-day profile')
    weekly.plot.scatter(x='temperature_2m_mean', y='load_mw', ax=axes[7], title='Temperature vs load')
    axes[8].axis('off')
    for ax in axes[:8]:
        ax.set_ylabel('Average load (MW)')
    axes[6].set_xlabel('Hour of day')
    axes[5].set_xlabel('Day of week')
    axes[7].set_xlabel('Weekly mean temperature (deg C)')

    for i, ax in enumerate(axes[:8]):
        if len(ax.lines) == 0 and len(ax.collections) == 0:
            raise RuntimeError(f'EDA subplot {i} is blank')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eda_overview.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    stl = STL(weekly['load_mw'].dropna(), period=52)
    res = stl.fit()
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), dpi=200)
    res.observed.plot(ax=axes[0], title='Observed')
    res.trend.plot(ax=axes[1], title='Trend')
    res.seasonal.plot(ax=axes[2], title='Seasonal')
    res.resid.plot(ax=axes[3], title='Residual')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'stl_decomposition.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    rolling = weekly['load_mw'].rolling(window=52).mean()
    std = weekly['load_mw'].rolling(window=52).std()
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    weekly['load_mw'].plot(ax=ax, alpha=0.5, label='Original')
    rolling.plot(ax=ax, label='Rolling mean 52W')
    std.plot(ax=ax, label='Rolling std 52W')
    ax.set_title('Rolling mean and standard deviation')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'rolling_stats.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
