from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'outputs' / 'tables'
FIGURES = ROOT / 'outputs' / 'figures'


REQUIRED_TABLES = [
    'outputs/tables/data_quality_summary.csv',
    'outputs/tables/train_test_split_summary.csv',
    'outputs/tables/stationarity_tests.csv',
    'outputs/tables/benchmark_metrics.csv',
    'outputs/tables/benchmark_forecasts.csv',
    'outputs/tables/sarima_grid_search.csv',
    'outputs/tables/sarima_metrics.csv',
    'outputs/tables/sarima_forecast.csv',
    'outputs/tables/sarima_residual_diagnostics.csv',
    'outputs/tables/sarimax_metrics.csv',
    'outputs/tables/sarimax_forecast.csv',
    'outputs/tables/feature_model_metrics.csv',
    'outputs/tables/feature_model_forecasts.csv',
    'outputs/tables/feature_importance.csv',
    'outputs/tables/lstm_metrics.csv',
    'outputs/tables/lstm_tuning_results.csv',
    'outputs/tables/lstm_hourly_forecast.csv',
    'outputs/tables/weekly_model_comparison.csv',
    'outputs/tables/forecast_interval_coverage.csv',
]

REQUIRED_FIGURES = [
    'outputs/figures/eda_overview.png',
    'outputs/figures/stl_decomposition.png',
    'outputs/figures/rolling_stats.png',
    'outputs/figures/original_weekly_load_acf_pacf.png',
    'outputs/figures/first_seasonal_differenced_weekly_load_acf_pacf.png',
    'outputs/figures/benchmark_forecasts.png',
    'outputs/figures/sarima_forecast.png',
    'outputs/figures/sarima_residual_overview.png',
    'outputs/figures/sarimax_forecast.png',
    'outputs/figures/feature_model_forecasts.png',
    'outputs/figures/lstm_training_history.png',
    'outputs/figures/lstm_first_week_forecast.png',
    'outputs/figures/lstm_first_month_forecast.png',
    'outputs/figures/weekly_model_rmse_comparison.png',
    'outputs/figures/weekly_model_mae_comparison.png',
    'outputs/figures/all_weekly_forecasts.png',
]

REQUIRED_PROJECT = [
    'models/hourly_lstm_model.keras',
    'README.md',
    'requirements.txt',
    'config.yaml',
    'run_pipeline.py',
    'report/report_evidence_guide.md',
    'src',
    'tests',
]


def report(name: str, ok: bool, details: str = '') -> bool:
    status = 'PASS' if ok else 'FAIL'
    print(f'{status}: {name}{(" - " + details) if details else ""}')
    return ok


def close(actual: float, expected: float, tolerance: float = 1e-3) -> bool:
    return abs(actual - expected) <= tolerance


def artifact_check() -> bool:
    rows = []
    ok = True
    for typ, paths in [('table', REQUIRED_TABLES), ('figure', REQUIRED_FIGURES), ('project', REQUIRED_PROJECT)]:
        for rel in paths:
            path = ROOT / rel
            exists = path.exists()
            size = path.stat().st_size if exists and path.is_file() else 0
            nonempty = exists and (path.is_dir() or size > 0)
            status = 'ok' if nonempty else 'missing_or_empty'
            rows.append({'path': rel, 'type': typ, 'exists': exists, 'size_bytes': size, 'status': status})
            ok = ok and nonempty
    pd.DataFrame(rows).to_csv(TABLES / 'final_artifact_check.csv', index=False)
    return report('required artifacts', ok, 'wrote outputs/tables/final_artifact_check.csv')


def metric_checks() -> bool:
    comparison = pd.read_csv(TABLES / 'weekly_model_comparison.csv')
    expected_models = {
        'Historical Mean',
        'Naive',
        'Seasonal Naive',
        'Drift',
        'SARIMA',
        'SARIMAX',
        'Random Forest',
        'Gradient Boosting',
    }
    required_cols = {
        'Model',
        'RMSE',
        'MAE',
        'MAPE',
        'sMAPE',
        'MASE',
        'RMSE_improvement_vs_seasonal_naive_percent',
        'MAE_improvement_vs_seasonal_naive_percent',
    }
    ok = set(comparison['Model']) == expected_models and required_cols.issubset(comparison.columns)
    rf = comparison.loc[comparison['Model'].eq('Random Forest')].iloc[0]
    gb = comparison.loc[comparison['Model'].eq('Gradient Boosting')].iloc[0]
    ok = ok and close(rf['RMSE'], 2650.601793) and close(rf['MAE'], 1980.456755)
    ok = ok and close(rf['RMSE_improvement_vs_seasonal_naive_percent'], 11.575, 0.02)
    ok = ok and close(rf['MAE_improvement_vs_seasonal_naive_percent'], 14.019, 0.02)
    ok = ok and close(gb['RMSE_improvement_vs_seasonal_naive_percent'], 4.456, 0.02)
    ok = ok and close(gb['MAE_improvement_vs_seasonal_naive_percent'], 2.821, 0.02)
    return report('weekly comparison metrics and improvements', ok)


def sarima_grid_check() -> bool:
    grid = pd.read_csv(TABLES / 'sarima_grid_search.csv')
    ok = len(grid) == 147
    ok = ok and sorted(grid['p'].dropna().unique().tolist()) == list(range(7))
    ok = ok and sorted(grid['d'].dropna().unique().tolist()) == list(range(3))
    ok = ok and sorted(grid['q'].dropna().unique().tolist()) == list(range(7))
    ok = ok and grid[['p', 'd', 'q']].drop_duplicates().shape[0] == 147
    valid = grid[grid['converged'].astype(str).str.lower().isin(['true', '1'])].sort_values('AIC')
    selected = valid.head(1).iloc[0]
    ok = ok and (int(selected['p']), int(selected['d']), int(selected['q']), int(selected['P']), int(selected['D']), int(selected['Q'])) == (0, 1, 6, 0, 1, 0)
    return report('SARIMA grid and selected order', ok, f'converged={int(grid["converged"].sum())}')


def forecast_checks() -> bool:
    weekly = pd.read_csv(ROOT / 'data' / 'processed' / 'weekly_load_temperature.csv', parse_dates=['date']).set_index('date').sort_index()
    expected_dates = weekly[weekly.index > weekly.index.max() - pd.DateOffset(years=2)].index
    benchmark = pd.read_csv(TABLES / 'benchmark_forecasts.csv')
    feature = pd.read_csv(TABLES / 'feature_model_forecasts.csv', parse_dates=['date'])
    ok = len(benchmark) == len(expected_dates)
    for model in ['SARIMA', 'SARIMAX']:
        df = pd.read_csv(TABLES / f'{model.lower()}_forecast.csv', parse_dates=['date'])
        dates = pd.DatetimeIndex(df['date'])
        ok = ok and dates.equals(expected_dates) and not dates.has_duplicates
        ok = ok and {'actual', 'forecast', 'lower_80', 'upper_80', 'lower_95', 'upper_95'}.issubset(df.columns)
        ok = ok and ((df['lower_95'] <= df['lower_80']) & (df['lower_80'] <= df['forecast']) & (df['forecast'] <= df['upper_80']) & (df['upper_80'] <= df['upper_95'])).all()
    for model_name, model_df in feature.groupby('Model'):
        dates = pd.DatetimeIndex(model_df['date'])
        ok = ok and dates.equals(expected_dates) and not dates.has_duplicates
    return report('forecast date alignment and interval ordering', ok)


def coverage_check() -> bool:
    coverage = pd.read_csv(TABLES / 'forecast_interval_coverage.csv')
    rows = []
    for model in ['SARIMA', 'SARIMAX']:
        df = pd.read_csv(TABLES / f'{model.lower()}_forecast.csv')
        for interval in ['80', '95']:
            covered = int(((df['actual'] >= df[f'lower_{interval}']) & (df['actual'] <= df[f'upper_{interval}'])).sum())
            rows.append((model, f'{interval}%', covered, len(df), covered / len(df)))
    recomputed = pd.DataFrame(rows, columns=['Model', 'interval', 'covered_count', 'total_count', 'coverage'])
    merged = coverage.merge(recomputed, on=['Model', 'interval'], suffixes=('', '_recomputed'))
    ok = len(merged) == 4
    ok = ok and (merged['covered_count'] == merged['covered_count_recomputed']).all()
    ok = ok and (merged['total_count'] == merged['total_count_recomputed']).all()
    ok = ok and (abs(merged['coverage'] - merged['coverage_recomputed']) < 1e-12).all()
    return report('forecast interval coverage recomputation', ok)


def stationarity_check() -> bool:
    df = pd.read_csv(TABLES / 'stationarity_tests.csv')
    required = {'series', 'ADF statistic', 'p-value', 'lags used', 'number of observations', '1% critical value', '5% critical value', '10% critical value', 'stationary at 5%'}
    ok = len(df) == 4 and required.issubset(df.columns)
    ok = ok and set(df['series']) == {'weekly', 'weekly_diff1', 'weekly_diff52', 'weekly_diff1_52'}
    return report('stationarity evidence table', ok)


def feature_importance_check() -> bool:
    df = pd.read_csv(TABLES / 'feature_importance.csv')
    ok = True
    for model, group in df.groupby('Model'):
        ok = ok and abs(group['importance'].sum() - 1.0) < 1e-6
    rf = df[df['Model'].eq('Random Forest')].head(5)
    expected = ['load_lag_52', 'temperature_2m_max', 'load_lag_1', 'week_of_year', 'temp_lag_52']
    ok = ok and rf['feature'].tolist() == expected
    return report('feature importances', ok)


def absolute_path_check() -> bool:
    bad = []
    local_home = '/home/' + 'reisecoder/'
    for path in [ROOT / 'README.md', ROOT / 'report' / 'report_evidence_guide.md']:
        text = path.read_text(encoding='utf-8')
        if local_home in text:
            bad.append(str(path.relative_to(ROOT)))
    return report('absolute local path search', not bad, ', '.join(bad))


def design_report() -> bool:
    print('INFO: Feature model forecast design: rolling one-step-ahead with observed prior test-period lagged load values.')
    print('INFO: LSTM forecast design: hourly one-step-ahead evaluation with chronological validation and shuffle=False.')
    return True


def main() -> int:
    checks = [
        artifact_check(),
        metric_checks(),
        sarima_grid_check(),
        forecast_checks(),
        coverage_check(),
        stationarity_check(),
        feature_importance_check(),
        absolute_path_check(),
        design_report(),
    ]
    ok = all(checks)
    print('FINAL VALIDATION:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
