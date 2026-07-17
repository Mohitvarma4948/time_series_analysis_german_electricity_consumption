from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import FIGURES_DIR, TABLES_DIR
from src.utils import calculate_metrics


def build_sarimax(logger: logging.Logger | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger('pipeline')
    weekly = pd.read_csv(Path('data/processed/weekly_load_temperature.csv'), parse_dates=['date'])
    weekly = weekly.set_index('date').sort_index()
    weekly = weekly[['load_mw', 'temperature_2m_mean', 'temperature_2m_min', 'temperature_2m_max', 'holiday_count']].dropna()
    weekly = weekly.asfreq('W-SUN')
    train_end = weekly.index.max() - pd.DateOffset(years=2)
    train = weekly[weekly.index <= train_end]
    test = weekly[weekly.index > train_end]

    exog_cols = ['temperature_2m_mean', 'holiday_count']
    X_train = train[exog_cols]
    X_test = test[exog_cols]
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    model = SARIMAX(train['load_mw'], exog=X_train_scaled, order=(1, 1, 1), seasonal_order=(0, 1, 0, 52), enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=100)
    forecast = fit.get_forecast(steps=len(test), exog=X_test_scaled)
    pred = forecast.predicted_mean
    ci_95 = forecast.conf_int(alpha=0.05)
    ci_80 = forecast.conf_int(alpha=0.20)
    pred.index = test.index
    ci_95.index = test.index
    ci_80.index = test.index
    metrics = calculate_metrics(test['load_mw'], pred, seasonal_period=52)
    metrics['Model'] = 'SARIMAX'
    metrics.to_csv(TABLES_DIR / 'sarimax_metrics.csv', index=False)
    forecast_df = pd.DataFrame({
        'date': test.index,
        'actual': test['load_mw'].values,
        'forecast': pred.values,
        'lower_80': ci_80.iloc[:, 0].values,
        'upper_80': ci_80.iloc[:, 1].values,
        'lower_95': ci_95.iloc[:, 0].values,
        'upper_95': ci_95.iloc[:, 1].values,
    })
    forecast_df.to_csv(TABLES_DIR / 'sarimax_forecast.csv', index=False)
    coeffs = pd.DataFrame({'name': fit.params.index, 'value': fit.params.values})
    coeffs.to_csv(TABLES_DIR / 'sarimax_coefficients.csv', index=False)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.plot(test.index, test['load_mw'], label='Actual')
    ax.plot(test.index, pred, label='Forecast')
    ax.fill_between(test.index, ci_95.iloc[:, 0], ci_95.iloc[:, 1], color='tab:orange', alpha=0.12, label='95% interval')
    ax.fill_between(test.index, ci_80.iloc[:, 0], ci_80.iloc[:, 1], color='tab:orange', alpha=0.22, label='80% interval')
    ax.set_title('SARIMAX weekly forecast')
    ax.set_ylabel('Average load (MW)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sarimax_forecast.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return metrics, forecast_df
