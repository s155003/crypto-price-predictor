# Crypto Price Predictor — Machine Learning

A Python machine learning project that fetches real cryptocurrency price data directly from Yahoo Finance and predicts the next day's closing price. Two models are trained and compared — Ridge Regression and Gradient Boosting — using over 30 engineered technical indicators as input features. The best model outputs a next-day price prediction alongside a bullish or bearish signal, and generates an 8-panel analysis chart.

---

## Overview

Cryptocurrency markets are notoriously volatile, making price prediction a challenging and well-studied problem in quantitative finance and machine learning. Rather than predicting exact prices with certainty, this project focuses on learning patterns in technical indicators — the same signals used by traders and analysts — and using them to forecast the next day's closing price with measurable accuracy.

The project is built to be flexible. Changing a single variable at the top of the script switches between Bitcoin, Ethereum, Solana, BNB, XRP, or any other valid Yahoo Finance ticker.

---

## Supported Coins

| Coin | Ticker |
|---|---|
| Bitcoin | BTC-USD |
| Ethereum | ETH-USD |
| Solana | SOL-USD |
| BNB | BNB-USD |
| XRP | XRP-USD |

Any valid Yahoo Finance crypto ticker works — just update `TICKER` at the top of the script.

---

## Models

| Model | Strengths |
|---|---|
| Ridge Regression | Fast, interpretable, strong regularized baseline |
| Gradient Boosting | Captures non-linear patterns, highest accuracy |

The best model is selected automatically based on lowest MAPE on the held-out test set.

---

## Feature Engineering

Over 30 features are engineered from raw OHLCV data before training:

### Price Returns
| Feature | Description |
|---|---|
| Return_1d | 1-day percentage return |
| Return_3d | 3-day percentage return |
| Return_7d | 7-day percentage return |

### Moving Averages
| Feature | Description |
|---|---|
| MA_7 / MA_14 / MA_30 | Simple moving averages over 7, 14, 30 days |
| EMA_12 / EMA_26 | Exponential moving averages |
| MACD | EMA_12 minus EMA_26 — momentum signal |

### Volatility & Risk
| Feature | Description |
|---|---|
| Volatility_7 | 7-day rolling standard deviation |
| Volatility_14 | 14-day rolling standard deviation |
| Price_range | Daily high minus low |
| Range_pct | Price range as percentage of close |

### Momentum
| Feature | Description |
|---|---|
| RSI | Relative Strength Index (14-day) — overbought/oversold signal |
| BB_upper / BB_lower | Bollinger Bands (20-day, 2 std dev) |
| BB_width | Band width relative to midline |
| BB_pos | Close price position within the bands |

### Volume
| Feature | Description |
|---|---|
| Volume_MA_7 | 7-day average volume |
| Volume_ratio | Current volume relative to 7-day average |

### Lagged Prices
Close prices from the past 14 days (Close_lag_1 through Close_lag_14) give the model explicit memory of recent price levels.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error — average dollar error per prediction |
| RMSE | Root Mean Squared Error — penalises large errors more |
| R² | Percentage of price variance explained by the model |
| MAPE | Mean Absolute Percentage Error — error relative to price level |
| Direction Accuracy | Percentage of days where predicted price movement direction was correct |

Direction Accuracy is particularly useful for crypto — even if exact dollar predictions are off, knowing whether the price will go up or down has real signal value.

---

## Output

Running the script prints a full model evaluation report and generates an 8-panel chart saved as `crypto_predictor.png`:

- **Full Price History** — entire dataset with train/test split marked
- **Predicted vs Actual** — both models overlaid on real test prices
- **Scatter Plot** — predicted vs actual for the best model
- **Residuals Over Time** — prediction errors plotted across the test period
- **Feature Importances** — top 15 features ranked by the best model
- **Model Comparison** — MAPE bar chart across both models
- **RSI Indicator** — recent RSI with overbought/oversold zones highlighted
- **Bollinger Bands** — recent price action within the bands

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
yfinance
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn yfinance
```

---

## Usage

```bash
python crypto_predictor.py
```

---

## Configuration

```python
TICKER     = "BTC-USD"     # Any Yahoo Finance crypto ticker
START_DATE = "2021-01-01"  # Historical data start
END_DATE   = "2024-12-31"  # Historical data end
LOOKBACK   = 14            # Number of lagged price features
TEST_SIZE  = 0.2           # 80/20 train/test split
```

---

## Example Terminal Output

```
Fetching BTC-USD from 2021-01-01 to 2024-12-31...
Loaded 1461 trading days

Training samples: 1168 | Test samples: 293
Features:         33

==================================================
  MODEL EVALUATION — BTC-USD
==================================================

  Ridge Regression
  ───────────────────────────────────────
  MAE:            $1,842.33
  RMSE:           $2,614.17
  R2:             0.9614
  MAPE:           4.82%
  Direction Acc:  54.3%

  Gradient Boosting
  ───────────────────────────────────────
  MAE:            $987.21
  RMSE:           $1,423.88
  R2:             0.9881
  MAPE:           2.31%
  Direction Acc:  61.7%

  Best model: Gradient Boosting (MAPE: 2.31%)

=============================================
  NEXT DAY PREDICTION — BTC-USD
=============================================
  Current price:    $98,432.00
  Predicted price:  $99,814.50
  Expected change:  +1,382.50 (+1.40%)
  Signal:           BULLISH
=============================================
```

---

## Extending the Project

- **Add an LSTM** — use TensorFlow/Keras to build a sequence model that processes price history as a time series rather than flat features
- **Multi-coin correlation** — add BTC dominance or ETH price as a feature when predicting altcoins
- **Sentiment features** — pull crypto Twitter/Reddit sentiment scores via API and include as input features
- **Backtesting** — use predicted buy/sell signals to simulate a trading strategy and calculate returns vs. buy-and-hold
- **Live predictions** — schedule the script to run daily and log predictions to a CSV for tracking over time

---

## Disclaimer

This project is built for educational purposes only. Cryptocurrency markets are highly volatile and unpredictable. Do not make real investment or trading decisions based on this model.
