from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import FIGURES_DIR, TABLES_DIR
from src.utils import calculate_metrics


def train_feature_models(logger: logging.Logger | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger('pipeline')
    dataset = pd.read_csv(Path('data/processed/weekly_feature_dataset.csv'), parse_dates=['date'])
    dataset = dataset.set_index('date').sort_index()
    target = dataset['load_mw']
    features = dataset.drop(columns=['load_mw'])
    tscv = TimeSeriesSplit(n_splits=3)
    rows = []
    models = {
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    }
    for name, model in models.items():
        for train_idx, test_idx in tscv.split(features):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            rows.append({'model': name, 'rmse': rmse, 'mae': mae})
    metrics = pd.DataFrame(rows).groupby('model').mean().reset_index()
    metrics.to_csv(TABLES_DIR / 'feature_model_cv_results.csv', index=False)

    train_end = dataset.index.max() - pd.DateOffset(years=2)
    X_train = features[features.index <= train_end]
    X_test = features[features.index > train_end]
    y_train = target[target.index <= train_end]
    y_test = target[target.index > train_end]
    metric_rows = []
    forecast_frames = []
    importance_frames = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = pd.Series(model.predict(X_test), index=y_test.index)
        model_metrics = calculate_metrics(y_test.reset_index(drop=True), pred.reset_index(drop=True), seasonal_period=52)
        model_metrics['Model'] = name
        metric_rows.append(model_metrics)
        forecast_frames.append(pd.DataFrame({'date': y_test.index, 'Model': name, 'actual': y_test.values, 'forecast': pred.values}))
        if hasattr(model, 'feature_importances_'):
            importance_frames.append(pd.DataFrame({
                'Model': name,
                'feature': features.columns,
                'importance': model.feature_importances_,
            }))
    metrics_df = pd.concat(metric_rows, ignore_index=True)
    metrics_df.to_csv(TABLES_DIR / 'feature_model_metrics.csv', index=False)
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    forecasts.to_csv(TABLES_DIR / 'feature_model_forecasts.csv', index=False)
    if importance_frames:
        importances = pd.concat(importance_frames, ignore_index=True)
        importances['model_order'] = importances['Model'].map({'Random Forest': 0, 'Gradient Boosting': 1}).fillna(99)
        importances = importances.sort_values(['model_order', 'importance'], ascending=[True, False])
        importances = importances.drop(columns=['model_order'])
        importances.to_csv(TABLES_DIR / 'feature_importance.csv', index=False)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
    for frame in forecast_frames:
        label = frame['Model'].iloc[0]
        ax.plot(frame['date'], frame['forecast'], label=label)
    ax.plot(y_test.index, y_test.values, label='Actual', color='black', linewidth=2)
    ax.set_title('Feature model weekly forecasts')
    ax.set_ylabel('Average load (MW)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_model_forecasts.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return metrics, metrics_df, forecasts, pd.DataFrame()
