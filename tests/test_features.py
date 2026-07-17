import pandas as pd
from pathlib import Path


def test_feature_dataset_has_expected_columns():
    df = pd.read_csv(Path('data/processed/weekly_feature_dataset.csv'), parse_dates=['date'])
    assert 'load_lag_52' in df.columns
    assert 'load_roll_mean_52' in df.columns
    assert 'temp_lag_52' in df.columns
