from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, TABLES_DIR


def create_weekly_comparison(logger: logging.Logger | None = None) -> pd.DataFrame:
    if logger is None:
        logger = logging.getLogger('pipeline')
    benchmark = pd.read_csv(TABLES_DIR / 'benchmark_metrics.csv')
    sarima = pd.read_csv(TABLES_DIR / 'sarima_metrics.csv')
    sarimax = pd.read_csv(TABLES_DIR / 'sarimax_metrics.csv')
    feature = pd.read_csv(TABLES_DIR / 'feature_model_metrics.csv') if (TABLES_DIR / 'feature_model_metrics.csv').exists() else pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'MAPE', 'sMAPE', 'MASE'])
    comparison = pd.concat([benchmark, sarima, sarimax, feature], ignore_index=True)
    comparison = comparison[['Model', 'RMSE', 'MAE', 'MAPE', 'sMAPE', 'MASE']]
    seasonal_row = comparison.loc[comparison['Model'].eq('Seasonal Naive')].iloc[0]
    comparison['RMSE_improvement_vs_seasonal_naive_percent'] = (
        (seasonal_row['RMSE'] - comparison['RMSE']) / seasonal_row['RMSE'] * 100
    )
    comparison['MAE_improvement_vs_seasonal_naive_percent'] = (
        (seasonal_row['MAE'] - comparison['MAE']) / seasonal_row['MAE'] * 100
    )
    comparison.to_csv(TABLES_DIR / 'weekly_model_comparison.csv', index=False)

    def plot_metric(metric: str, output_name: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ordered = comparison.sort_values(metric)
        bars = ax.bar(ordered['Model'], ordered[metric])
        ax.axhline(seasonal_row[metric], color='tab:red', linestyle='--', linewidth=1.2, label='Seasonal naive')
        ax.set_title(f'Weekly {metric} comparison')
        ax.set_ylabel(f'{metric} (MW)')
        ax.tick_params(axis='x', rotation=35)
        ax.legend()
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f'{value:.1f}',
                ha='center',
                va='bottom',
                fontsize=7,
            )
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches='tight')
        plt.close(fig)

    plot_metric('RMSE', 'weekly_model_rmse_comparison.png')
    plot_metric('MAE', 'weekly_model_mae_comparison.png')

    weekly = pd.read_csv(Path('data/processed/weekly_load_temperature.csv'), parse_dates=['date'])
    weekly = weekly.set_index('date').sort_index()
    test_dates = weekly[weekly.index > weekly.index.max() - pd.DateOffset(years=2)].index
    benchmark_forecasts = pd.read_csv(TABLES_DIR / 'benchmark_forecasts.csv')
    sarima_forecast = pd.read_csv(TABLES_DIR / 'sarima_forecast.csv', parse_dates=['date'])
    sarimax_forecast = pd.read_csv(TABLES_DIR / 'sarimax_forecast.csv', parse_dates=['date'])
    feature_forecasts = pd.read_csv(TABLES_DIR / 'feature_model_forecasts.csv', parse_dates=['date'])

    expected_dates = pd.DatetimeIndex(test_dates)
    if len(benchmark_forecasts) != len(expected_dates):
        raise ValueError('Benchmark forecast length does not match weekly test dates.')
    for name, forecast_df in {'SARIMA': sarima_forecast, 'SARIMAX': sarimax_forecast}.items():
        dates = pd.DatetimeIndex(forecast_df['date'])
        if not dates.equals(expected_dates):
            raise ValueError(f'{name} forecast dates do not align with weekly test dates.')
        if dates.has_duplicates:
            raise ValueError(f'{name} forecast contains duplicated dates.')
    for model_name, model_df in feature_forecasts.groupby('Model'):
        dates = pd.DatetimeIndex(model_df['date'])
        if not dates.equals(expected_dates):
            raise ValueError(f'{model_name} forecast dates do not align with weekly test dates.')
        if dates.has_duplicates:
            raise ValueError(f'{model_name} forecast contains duplicated dates.')

    split_rows = []
    weekly_train = weekly[weekly.index <= weekly.index.max() - pd.DateOffset(years=2)]
    weekly_test = weekly[weekly.index > weekly.index.max() - pd.DateOffset(years=2)]
    split_rows.append({
        'dataset': 'weekly_models',
        'train_start': weekly_train.index.min(),
        'train_end': weekly_train.index.max(),
        'test_start': weekly_test.index.min(),
        'test_end': weekly_test.index.max(),
        'train_rows': len(weekly_train),
        'test_rows': len(weekly_test),
    })
    if Path('data/processed/weekly_feature_dataset.csv').exists():
        feature_data = pd.read_csv(Path('data/processed/weekly_feature_dataset.csv'), parse_dates=['date']).set_index('date').sort_index()
        feature_train = feature_data[feature_data.index <= feature_data.index.max() - pd.DateOffset(years=2)]
        feature_test = feature_data[feature_data.index > feature_data.index.max() - pd.DateOffset(years=2)]
        split_rows.append({
            'dataset': 'feature_models',
            'train_start': feature_train.index.min(),
            'train_end': feature_train.index.max(),
            'test_start': feature_test.index.min(),
            'test_end': feature_test.index.max(),
            'train_rows': len(feature_train),
            'test_rows': len(feature_test),
        })
    if Path('data/processed/hourly_load.csv').exists():
        hourly = pd.read_csv(Path('data/processed/hourly_load.csv'), parse_dates=['utc_timestamp']).set_index('utc_timestamp').sort_index()
        hourly_train = hourly[hourly.index <= hourly.index.max() - pd.DateOffset(years=2)]
        hourly_test = hourly[hourly.index > hourly.index.max() - pd.DateOffset(years=2)]
        split_rows.append({
            'dataset': 'lstm_hourly',
            'train_start': hourly_train.index.min(),
            'train_end': hourly_train.index.max(),
            'test_start': hourly_test.index.min(),
            'test_end': hourly_test.index.max(),
            'train_rows': len(hourly_train),
            'test_rows': len(hourly_test),
        })
    pd.DataFrame(split_rows).to_csv(TABLES_DIR / 'train_test_split_summary.csv', index=False)

    coverage_rows = []
    for model_name, forecast_df in {'SARIMA': sarima_forecast, 'SARIMAX': sarimax_forecast}.items():
        for interval in ['80', '95']:
            lower = forecast_df[f'lower_{interval}']
            upper = forecast_df[f'upper_{interval}']
            actual = forecast_df['actual']
            coverage_rows.append({
                'Model': model_name,
                'interval': f'{interval}%',
                'covered_count': int(((actual >= lower) & (actual <= upper)).sum()),
                'total_count': len(forecast_df),
                'coverage': float(((actual >= lower) & (actual <= upper)).mean()),
            })
    pd.DataFrame(coverage_rows).to_csv(TABLES_DIR / 'forecast_interval_coverage.csv', index=False)

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.plot(test_dates, benchmark_forecasts['actual'], color='black', linewidth=2.3, label='Actual')
    benchmark_styles = {
        'Historical Mean': ':',
        'Naive': '-.',
        'Seasonal Naive': '--',
        'Drift': ':',
    }
    for col, style in benchmark_styles.items():
        ax.plot(test_dates, benchmark_forecasts[col], linestyle=style, linewidth=1.2, label=col)
    ax.plot(sarima_forecast['date'], sarima_forecast['forecast'], linewidth=1.2, label='SARIMA')
    ax.plot(sarimax_forecast['date'], sarimax_forecast['forecast'], linewidth=1.2, label='SARIMAX')
    for model_name, model_df in feature_forecasts.groupby('Model'):
        ax.plot(model_df['date'], model_df['forecast'], linewidth=1.4, label=model_name)
    ax.set_title('Weekly forecasts across evaluated models')
    ax.set_ylabel('Average load (MW)')
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'all_weekly_forecasts.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    if (TABLES_DIR / 'lstm_hourly_forecast.csv').exists() and (TABLES_DIR / 'lstm_metrics.csv').exists():
        lstm_forecast = pd.read_csv(TABLES_DIR / 'lstm_hourly_forecast.csv', parse_dates=['date'])
        lstm_metrics = pd.read_csv(TABLES_DIR / 'lstm_metrics.csv')
        tuning = lstm_metrics.copy()
        tuning['lookback_hours'] = 168
        tuning['validation_design'] = 'chronological final 10% of training sequences'
        tuning['shuffle'] = False
        tuning.to_csv(TABLES_DIR / 'lstm_tuning_results.csv', index=False)
        for label, rows, filename in [
            ('First week LSTM hourly forecast', 24 * 7, 'lstm_first_week_forecast.png'),
            ('First month LSTM hourly forecast', 24 * 30, 'lstm_first_month_forecast.png'),
        ]:
            subset = lstm_forecast.head(rows)
            fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
            ax.plot(subset['date'], subset['actual'], label='Actual', color='black', linewidth=1.8)
            ax.plot(subset['date'], subset['forecast'], label='Forecast', linewidth=1.2)
            ax.set_title(label)
            ax.set_ylabel('Load (MW)')
            ax.legend()
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / filename, dpi=200, bbox_inches='tight')
            plt.close(fig)
    return comparison
