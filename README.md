# Machine Learning Experiments

A curated set of personal machine-learning experiments using **Python**, **TensorFlow/Keras**, **scikit-learn**, and **Google Colab**.

The project combines a CIFAR-10 image-classification comparison with recovered **2023 stock-price modeling notebooks**. These are learning experiments, not production predictions or a published research benchmark.

## Projects

| Area | Location | What it demonstrates |
| --- | --- | --- |
| CIFAR-10 classification | [cifar10.py](cifar10.py) | CNN, SVM, Random Forest and Bernoulli RBM + logistic regression |
| Stock closing-price forecasting | [stock_forecasting/](stock_forecasting/) | Leakage-aware chronological split, sliding windows, LSTM training and naïve baseline |
| Original LSTM experiment (Jan 2023) | [notebooks/stock_lstm_original_2023.ipynb](notebooks/stock_lstm_original_2023.ipynb) | Historical Keras training code and saved notebook outputs |
| Original Prophet experiment (Jan 2023) | [notebooks/stock_prophet_original_2023.ipynb](notebooks/stock_prophet_original_2023.ipynb) | Early Prophet and Yahoo Finance exploration |
| Original Weka experiment (Jan 2023) | [notebooks/stock_weka_original_2023.ipynb](notebooks/stock_weka_original_2023.ipynb) | Early Weka linear-regression exploration |

## Getting started

Python 3.10–3.12 recommended. TensorFlow is a comparatively large installation; use a supported Python environment.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

# Controlled subset of CIFAR-10: fewer samples for practical local experimentation
python cifar10.py --model cnn --train-size 4000 --test-size 1000 --epochs 3
python cifar10.py --model svm --train-size 1500 --test-size 500

# Create clearly labeled synthetic demo prices (not real market data)
python -m stock_forecasting.generate_demo_data --output data/demo_synthetic.csv

# Train a next-day close-price model and compare against last-close baseline
python -m stock_forecasting.train_lstm --csv data/demo_synthetic.csv --epochs 3

# Run tests (excluding expensive training runs)
python -m unittest discover -s tests -v
```

You may replace the synthetic demo CSV with **historical prices you are authorized to use**. The stock script expects `Close` and optionally `Date` columns sorted chronologically or sortable by `Date`.

## Critical historical limitations

The original 2023 LSTM notebook is preserved **as-is** for provenance. It scales the full series before train/test splitting and uses the same day's closing price as an input to predict that closing price. That is **information leakage**, so its low reported MSE is *not* reliable evidence of future forecasting accuracy. The new `stock_forecasting/` demonstration predicts the **next close** from preceding closing prices, fits its scaler only on training history and reports a last-close baseline.

The original Prophet and Weka notebooks also contain incomplete/obsolete steps and have not been reproduced end to end. They are archival evidence of learning, not verified working products. The earlier CIFAR-10 script is retained in `main.py` as historical reference; `cifar10.py` is a better-controlled baseline.

## Results

No new benchmark accuracy or financial performance claim is made here. Running the scripts prints local metrics; results depend on hardware, random seed, data selection and hyperparameters. In particular, predicting prices should **never** be taken as financial advice.

## Roadmap

- Record dataset versions, environment and completed training runs in a results table
- Add repeated evaluation across seeds and more conservative time-series backtesting
- Include confusion matrices and classwise scores for CIFAR-10
- Add clearer model-comparison plots and reproducible experiment artifacts

The original school-era code remains accessible in Git history and the notebook archive.
