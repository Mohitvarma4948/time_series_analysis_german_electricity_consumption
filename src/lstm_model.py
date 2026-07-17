from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import CONFIG, FIGURES_DIR, MODELS_DIR, TABLES_DIR
from src.utils import calculate_metrics


def build_lstm(logger: logging.Logger | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger('pipeline')
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
        from tensorflow.keras.models import Sequential
    except Exception as exc:
        raise RuntimeError(
            'TensorFlow is required only for the LSTM stage. Install a working '
            'tensorflow-cpu package, or run the other pipeline stages separately.'
        ) from exc

    hourly = pd.read_csv(Path('data/processed/hourly_load.csv'), parse_dates=['utc_timestamp'])
    hourly = hourly.set_index('utc_timestamp').sort_index()
    hourly = hourly[['load_mw']].dropna()
    train_end = hourly.index.max() - pd.DateOffset(years=2)
    train = hourly[hourly.index <= train_end]
    test = hourly[hourly.index > train_end]
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[['load_mw']])
    test_scaled = scaler.transform(test[['load_mw']])

    def create_sequences(values: np.ndarray, lookback: int, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(values) - lookback - horizon + 1):
            X.append(values[i:i + lookback])
            y.append(values[i + lookback + horizon - 1])
        return np.array(X), np.array(y)

    lookback = int(CONFIG['settings'].get('lstm_lookback', 168))
    epochs = int(CONFIG['settings'].get('lstm_epochs', 10))
    X_train, y_train = create_sequences(train_scaled, lookback=lookback)
    X_test, y_test = create_sequences(test_scaled, lookback=lookback)
    val_size = max(1, int(len(X_train) * 0.1))
    X_fit, X_val = X_train[:-val_size], X_train[-val_size:]
    y_fit, y_val = y_train[:-val_size], y_train[-val_size:]

    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(str(MODELS_DIR / 'hourly_lstm_model.keras'), save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    history = model.fit(
        X_fit.reshape(X_fit.shape[0], X_fit.shape[1], 1),
        y_fit,
        validation_data=(X_val.reshape(X_val.shape[0], X_val.shape[1], 1), y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,
    )

    pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), verbose=0).ravel()
    pred = scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
    actual = test[['load_mw']].iloc[lookback:].values.ravel()
    actual_dates = test.index[lookback:]
    metrics = calculate_metrics(pd.Series(actual), pd.Series(pred), seasonal_period=168)
    metrics['Model'] = 'LSTM'
    metrics.to_csv(TABLES_DIR / 'lstm_metrics.csv', index=False)
    forecast_df = pd.DataFrame({'date': actual_dates, 'actual': actual, 'forecast': pred})
    forecast_df.to_csv(TABLES_DIR / 'lstm_hourly_forecast.csv', index=False)
    plt.figure(figsize=(10, 4), dpi=200)
    epochs_seen = np.arange(1, len(history.history['loss']) + 1)
    plt.plot(epochs_seen, history.history['loss'], marker='o', label='Train loss')
    plt.plot(epochs_seen, history.history['val_loss'], marker='o', label='Validation loss')
    plt.title('LSTM training history')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.xticks(epochs_seen)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'lstm_training_history.png', dpi=200, bbox_inches='tight')
    plt.close()
    return metrics, forecast_df
