from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'config.yaml'

with CONFIG_PATH.open('r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

PROJECT_NAME = CONFIG['project']['name']
SEED = CONFIG['project']['seed']


def resolve_data_dir() -> Path:
    project_raw = ROOT / 'data' / 'raw'
    if (project_raw / 'time_series_60min_singleindex.csv').exists() and (project_raw / 'berlin_temperature_daily.csv').exists():
        return project_raw
    workspace_raw = ROOT.parent / 'data' / 'raw'
    if (workspace_raw / 'time_series_60min_singleindex.csv').exists() and (workspace_raw / 'berlin_temperature_daily.csv').exists():
        return workspace_raw
    return project_raw

RAW_DATA_DIR = resolve_data_dir()
PROCESSED_DATA_DIR = ROOT / CONFIG['paths']['processed_data']
OUTPUTS_DIR = ROOT / CONFIG['paths']['outputs']
FIGURES_DIR = ROOT / CONFIG['paths']['figures']
TABLES_DIR = ROOT / CONFIG['paths']['tables']
LOGS_DIR = ROOT / CONFIG['paths']['logs']
MODELS_DIR = ROOT / CONFIG['paths']['models']
REPORT_DIR = ROOT / CONFIG['paths']['report']

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR, MODELS_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

ELECTRICITY_FILE = RAW_DATA_DIR / 'time_series_60min_singleindex.csv'
TEMPERATURE_FILE = RAW_DATA_DIR / 'berlin_temperature_daily.csv'

LOAD_COLUMN = 'DE_load_actual_entsoe_transparency'
TIMESTAMP_COLUMN = 'utc_timestamp'
LOAD_OUTPUT = PROCESSED_DATA_DIR / 'hourly_load.csv'
DAILY_OUTPUT = PROCESSED_DATA_DIR / 'daily_load.csv'
WEEKLY_OUTPUT = PROCESSED_DATA_DIR / 'weekly_load.csv'
WEEKLY_TEMP_OUTPUT = PROCESSED_DATA_DIR / 'weekly_temperature.csv'
WEEKLY_MERGED_OUTPUT = PROCESSED_DATA_DIR / 'weekly_load_temperature.csv'
FEATURE_DATA_OUTPUT = PROCESSED_DATA_DIR / 'weekly_feature_dataset.csv'
