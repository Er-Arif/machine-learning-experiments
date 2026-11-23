"""Time-ordered close-price preparation for next-step forecasting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_close_prices(path: str | Path) -> np.ndarray:
    frame = pd.read_csv(path)
    columns = {column.casefold(): column for column in frame.columns}
    if "close" not in columns:
        raise ValueError("CSV must contain a Close column")
    if "date" in columns:
        dates = pd.to_datetime(frame[columns["date"]], errors="coerce")
        if dates.isna().any():
            raise ValueError("CSV contains invalid dates")
        frame = frame.assign(_date=dates).sort_values("_date", kind="stable")
    close = pd.to_numeric(frame[columns["close"]], errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    prices = close.to_numpy(dtype=np.float64)
    if len(prices) < 30:
        raise ValueError("At least 30 valid closing prices are required")
    return prices


def prepare_sequences(prices, *, lookback: int = 20, train_fraction: float = 0.8):
    """Fit scaler on training history; predict each next close from past closes."""
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("prices must be a finite one-dimensional series")
    if lookback < 2 or not 0.5 <= train_fraction < 1:
        raise ValueError("Need lookback >= 2 and 0.5 <= train_fraction < 1")
    split = int(len(values) * train_fraction)
    if split <= lookback or split >= len(values):
        raise ValueError("Not enough training/test prices for this lookback")

    scaler = MinMaxScaler()
    scaler.fit(values[:split].reshape(-1, 1))  # No test-set information
    scaled = scaler.transform(values.reshape(-1, 1)).ravel()
    sequences = np.stack([scaled[i - lookback:i] for i in range(lookback, len(values))])
    targets = scaled[lookback:]
    target_positions = np.arange(lookback, len(values))
    training = target_positions < split

    x_train = sequences[training, :, None].astype(np.float32)
    y_train = targets[training].astype(np.float32)
    x_test = sequences[~training, :, None].astype(np.float32)
    y_test = targets[~training].astype(np.float32)
    return x_train, y_train, x_test, y_test, scaler
