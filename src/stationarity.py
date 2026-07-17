from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf

from src.config import FIGURES_DIR, TABLES_DIR


def run_stationarity_analysis(logger: logging.Logger | None = None) -> pd.DataFrame:
    if logger is None:
        logger = logging.getLogger('pipeline')
    weekly = pd.read_csv(Path('data/processed/weekly_load_temperature.csv'), parse_dates=['date'])
    weekly = weekly.set_index('date').sort_index()
    series = weekly['load_mw'].dropna()
    results = []
    for label, data in {
        'weekly': series,
        'weekly_diff1': series.diff().dropna(),
        'weekly_diff52': series.diff(52).dropna(),
        'weekly_diff1_52': series.diff().diff(52).dropna(),
    }.items():
        stat, pval, lags, nobs, critical_values, _icbest = adfuller(data)
        results.append({
            'series': label,
            'ADF statistic': stat,
            'p-value': pval,
            'lags used': lags,
            'number of observations': nobs,
            '1% critical value': critical_values['1%'],
            '5% critical value': critical_values['5%'],
            '10% critical value': critical_values['10%'],
            'stationary at 5%': pval < 0.05,
        })
    table = pd.DataFrame(results)
    table.to_csv(TABLES_DIR / 'stationarity_tests.csv', index=False)

    plot_data = {
        'Original weekly load': series,
        'First + seasonal differenced weekly load': series.diff().diff(52).dropna(),
    }
    for label, data in plot_data.items():
        lags = min(60, len(data) // 2 - 1)
        acf_values = acf(data, nlags=lags, fft=True)
        pacf_values = pacf(data, nlags=lags)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200)
        axes[0].bar(range(len(acf_values)), acf_values)
        axes[0].set_title(f'ACF: {label}')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('Autocorrelation')
        axes[1].bar(range(len(pacf_values)), pacf_values)
        axes[1].set_title(f'PACF: {label}')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('Partial autocorrelation')
        plt.tight_layout()
        filename = label.lower().replace(' + ', '_').replace(' ', '_').replace(':', '')
        plt.savefig(FIGURES_DIR / f'{filename}_acf_pacf.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
    return table
