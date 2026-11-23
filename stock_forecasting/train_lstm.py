"""Train a next-day LSTM, evaluate against a last-close baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from stock_forecasting.data import load_close_prices, prepare_sequences


def evaluate(prices, *, lookback: int, epochs: int, output_dir: Path, seed: int):
    import tensorflow as tf

    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    x_train, y_train, x_test, y_test, scaler = prepare_sequences(
        prices, lookback=lookback
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input((lookback, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=2,
              shuffle=False)
    predicted_scaled = model.predict(x_test, verbose=0).reshape(-1, 1)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    predicted = scaler.inverse_transform(predicted_scaled).ravel()
    naive = scaler.inverse_transform(x_test[:, -1, 0].reshape(-1, 1)).ravel()

    def metrics(estimates):
        return {
            "mae": float(mean_absolute_error(actual, estimates)),
            "rmse": float(np.sqrt(mean_squared_error(actual, estimates))),
        }

    results = {
        "experiment": "next-day close from previous closes",
        "test_samples": int(len(actual)),
        "lookback": lookback,
        "lstm": metrics(predicted),
        "last_close_baseline": metrics(naive),
        "notes": "Chronological holdout; scaler fitted only on training history",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "lstm_model.keras")
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="CSV with Close column")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("output/stock_lstm"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    prices = load_close_prices(args.csv)
    evaluate(prices, lookback=args.lookback, epochs=args.epochs,
             output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
