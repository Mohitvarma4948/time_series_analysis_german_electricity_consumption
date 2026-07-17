from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import ROOT
from src.data_preparation import inspect_raw_data, prepare_data
from src.exploratory_analysis import plot_series
from src.stationarity import run_stationarity_analysis
from src.benchmarks import evaluate_benchmarks
from src.sarima_model import search_sarima
from src.sarimax_model import build_sarimax
from src.feature_engineering import build_feature_dataset
from src.feature_models import train_feature_models
from src.lstm_model import build_lstm
from src.evaluation import create_weekly_comparison
from src.utils import setup_logger, set_seed


STAGES = ['inspect', 'data', 'eda', 'benchmarks', 'sarima', 'sarimax', 'features', 'lstm', 'evaluate', 'all']


def run_stage(stage: str) -> None:
    logger = setup_logger('pipeline', str(ROOT / 'outputs' / 'logs' / 'pipeline.log'))
    set_seed()
    if stage in {'inspect', 'all'}:
        logger.info('Starting inspection stage')
        inspect_raw_data()
    if stage in {'data', 'all'}:
        logger.info('Starting data preparation stage')
        prepare_data(logger)
    if stage in {'eda', 'all'}:
        logger.info('Starting exploratory analysis stage')
        plot_series(logger)
    if stage in {'benchmarks', 'all'}:
        logger.info('Starting benchmarks stage')
        evaluate_benchmarks(logger)
    if stage in {'sarima', 'all'}:
        logger.info('Starting SARIMA stage')
        search_sarima(logger)
    if stage in {'sarimax', 'all'}:
        logger.info('Starting SARIMAX stage')
        build_sarimax(logger)
    if stage in {'features', 'all'}:
        logger.info('Starting feature engineering and feature models stage')
        build_feature_dataset(logger)
        train_feature_models(logger)
    if stage in {'lstm', 'all'}:
        logger.info('Starting LSTM stage')
        build_lstm(logger)
    if stage in {'evaluate', 'all'}:
        logger.info('Starting evaluation stage')
        create_weekly_comparison(logger)
        run_stationarity_analysis(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the German electricity demand forecasting pipeline')
    parser.add_argument('--stage', choices=STAGES, default='all')
    args = parser.parse_args()
    run_stage(args.stage)


if __name__ == '__main__':
    main()
