import pandas as pd
from pathlib import Path

from src.utils import calculate_metrics


def test_metrics_return_expected_columns():
    actual = pd.Series([1.0, 2.0, 3.0])
    forecast = pd.Series([1.0, 2.0, 4.0])
    metrics = calculate_metrics(actual, forecast)
    assert set(metrics.columns) == {'RMSE', 'MAE', 'MAPE', 'sMAPE', 'MASE'}


def test_forecast_files_exist():
    assert Path('outputs/tables/benchmark_metrics.csv').exists()
