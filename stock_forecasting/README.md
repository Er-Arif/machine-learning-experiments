# LSTM Stock Close Forecasting (Educational)

A reproducible follow-up to the original **January 2023 Google Colab experiment**, developed to make the method easier to review and to fix information leakage.

- [Original 2023 LSTM notebook](../notebooks/stock_lstm_original_2023.ipynb) contains the historic code and run outputs, preserved without rewriting them.
- [data.py](data.py) prepares a chronological holdout and lookback windows.
- [train_lstm.py](train_lstm.py) trains a small Keras LSTM model and compares it against a naïve last-close forecast.
- [generate_demo_data.py](generate_demo_data.py) generates synthetic demo prices solely to exercise the pipeline.

## Why a newer implementation?

The original notebook used both today's Open and today's Close to estimate today's Close, and fitted a scaler on the full dataset before splitting. That does not demonstrate accurate forecasting of unseen future prices.

The improved example:

1. Sorts records by Date, when supplied.
2. Reserves the last 20% of observations for the test set.
3. Fits `MinMaxScaler` on the training period only.
4. Predicts each day's Close using only preceding Close prices.
5. Compares model MAE/RMSE in original price units with predicting the previous close.

## Run

From the **repository root**:

```bash
python -m pip install -r requirements.txt
python -m stock_forecasting.generate_demo_data --output data/demo_synthetic.csv
python -m stock_forecasting.train_lstm --csv data/demo_synthetic.csv --lookback 20 --epochs 3
```

You may replace the synthetic CSV with your own legitimate historical price dataset. Columns: `Close` (required) and `Date` (recommended). The scripts save a trained model and `metrics.json` locally in `output/stock_lstm/`; these are **not** committed.

## Interpretation and limits

A single chronological holdout is only an educational baseline. Real financial backtesting needs repeated walk-forward evaluation, transaction-cost and regime considerations, and comparison to several baselines. Results are not investment advice. No new performance score is claimed without running the experiment.
