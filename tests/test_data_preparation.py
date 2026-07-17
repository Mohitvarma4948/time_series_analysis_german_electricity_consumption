import pandas as pd
from pathlib import Path

from src.data_preparation import prepare_data


def test_processed_data_has_no_duplicates():
    hourly = pd.read_csv(Path('data/processed/hourly_load.csv'), parse_dates=['utc_timestamp'])
    assert hourly['utc_timestamp'].is_unique


def test_weekly_aggregation_shape():
    weekly = pd.read_csv(Path('data/processed/weekly_load.csv'), parse_dates=['date'])
    assert len(weekly) > 0
