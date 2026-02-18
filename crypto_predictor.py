import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")


TICKER      = "BTC-USD"
START_DATE  = "2021-01-01"
END_DATE    = "2024-12-31"
LOOKBACK    = 14
TEST_SIZE   = 0.2
RANDOM_SEED = 42


CRYPTO_TICKERS = {
    "Bitcoin":  "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana":   "SOL-USD",
    "BNB":      "BNB-USD",
    "XRP":      "XRP-USD",
}


def fetch_data(ticker, start, end):
    print(f"Fetching {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df.dropna(inplace=True)
    print(f"Loaded {len(df)} trading days")
    return df


def add_features(df):
    df = df.copy()
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    df["Return_1d"]    = close.pct_change(1)
    df["Return_3d"]    = close.pct_change(3)
    df["Return_7d"]    = close.pct_change(7)
    df["MA_7"]         = close.rolling(7).mean()
    df["MA_14"]        = close.rolling(14).mean()
    df["MA_30"]        = close.rolling(30).mean()
    df["EMA_12"]       = close.ewm(span=12).mean()
    df["EMA_26"]       = close.ewm(span=26).mean()
    df["MACD"]         = df["EMA_12"] - df["EMA_26"]
    df["Volatility_7"] = close.rolling(7).std()
    df["Volatility_14"]= close.rolling(14).std()
    df["Volume_MA_7"]  = volume.rolling(7).mean()
    df["Volume_ratio"] = volume / (df["Volume_MA_7"] + 1e-6)
    df["Price_range"]  = df["High"].squeeze() - df["Low"].squeeze()
    df["Range_pct"]    = df["Price_range"] / (close + 1e-6)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-6)
    df["RSI"] = 100 - (100 / (1 + rs))

    bb_mid         = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / (bb_mid + 1e-6)
    df["BB_pos"]   = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-6)

    for lag in range(1, LOOKBACK + 1):
        df[f"Close_lag_{lag}"] = close.shift(lag)

    df["Target"] = close.shift(-1)
    df.dropna(inplace=True)
    return df


def get_feature_cols(df):
    exclude = {"Target", "Open", "High", "Low", "Close", "Volume"}
    return [c for c in df.columns if c not in exclude]


def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
    dirn = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
    print(f"\n  {name}")
    print(f"  {'─'*35}")
    print(f"  MAE:            ${mae:,.2f}")
    print(f"  RMSE:           ${rmse:,.2f}")
    print(f"  R2:             {r2:.4f}")
    print(f"  MAPE:           {mape:.2f}%")
    print(f"  Direction Acc:  {dirn:.1f}%")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "direction": dirn}


def plot_results(df, y_test, preds_dict, test_dates, feature_cols, importances, scores):
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(f"{TICKER} Price Predictor — ML Analysis", fontsize=16, fontweight="bold")

    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(df.index, df["Close"].squeeze(), color="#1f77b4", linewidth=1, label="Close Price")
    ax1.axvline(test_dates[0], color="red", linestyle="--", alpha=0.7, label="Train/Test Split")
    ax1.set_title("Full Price History")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2 = fig.add_subplot(4, 2, 2)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (name, preds), color in zip(preds_dict.items(), colors):
        ax2.plot(test_dates, preds, linewidth=1, label=f"{name}", color=color, alpha=0.8)
    ax2.plot(test_dates, y_test, color="black", linewidth=1.2, label="Actual", alpha=0.9)
    ax2.set_title("Test Period: Predicted vs Actual")
    ax2.set_ylabel("Price (USD)")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    best_name  = min(scores, key=lambda k: scores[k]["mape"])
    best_preds = preds_dict[best_name]

    ax3 = fig.add_subplot(4, 2, 3)
    ax3.scatter(y_test, best_preds, alpha=0.3, s=8, color="#2ca02c")
    mn, mx = min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())
    ax3.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect")
    ax3.set_xlabel("Actual Price ($)")
    ax3.set_ylabel("Predicted Price ($)")
    ax3.set_title(f"Predicted vs Actual ({best_name})")
    ax3.legend()

    ax4 = fig.add_subplot(4, 2, 4)
    residuals = y_test - best_preds
    ax4.plot(test_dates, residuals, color="#d62728", linewidth=0.8, alpha=0.8)
    ax4.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax4.fill_between(test_dates, residuals, 0, alpha=0.2, color="#d62728")
    ax4.set_title(f"Residuals Over Time ({best_name})")
    ax4.set_ylabel("Residual ($)")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax5 = fig.add_subplot(4, 2, 5)
    sorted_idx   = np.argsort(importances)[::-1][:15]
    sorted_names = [feature_cols[i] for i in sorted_idx]
    sorted_imp   = importances[sorted_idx]
    bar_colors   = ["#d62728" if i == 0 else "#1f77b4" for i in range(len(sorted_imp))]
    ax5.barh(sorted_names[::-1], sorted_imp[::-1], color=bar_colors[::-1])
    ax5.set_xlabel("Importance")
    ax5.set_title(f"Top 15 Feature Importances ({best_name})")

    ax6 = fig.add_subplot(4, 2, 6)
    model_names = list(scores.keys())
    mapes = [scores[n]["mape"] for n in model_names]
    bar_c = ["#2ca02c" if n == best_name else "#1f77b4" for n in model_names]
    bars  = ax6.bar(model_names, mapes, color=bar_c)
    ax6.set_ylabel("MAPE (%)")
    ax6.set_title("Model Comparison — MAPE (lower is better)")
    for bar, v in zip(bars, mapes):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{v:.2f}%", ha="center", fontsize=9)

    ax7 = fig.add_subplot(4, 2, 7)
    rsi_vals = df["RSI"].squeeze()
    ax7.plot(df.index[-len(y_test)*2:], rsi_vals[-len(y_test)*2:], color="#ff7f0e", linewidth=1)
    ax7.axhline(70, color="red",   linestyle="--", linewidth=1, label="Overbought (70)")
    ax7.axhline(30, color="green", linestyle="--", linewidth=1, label="Oversold (30)")
    ax7.fill_between(df.index[-len(y_test)*2:], rsi_vals[-len(y_test)*2:], 70,
                     where=rsi_vals[-len(y_test)*2:] >= 70, alpha=0.2, color="red")
    ax7.fill_between(df.index[-len(y_test)*2:], rsi_vals[-len(y_test)*2:], 30,
                     where=rsi_vals[-len(y_test)*2:] <= 30, alpha=0.2, color="green")
    ax7.set_title("RSI Indicator (Recent)")
    ax7.set_ylabel("RSI")
    ax7.set_ylim(0, 100)
    ax7.legend(fontsize=8)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax8 = fig.add_subplot(4, 2, 8)
    close_recent = df["Close"].squeeze()[-len(y_test)*2:]
    bb_up        = df["BB_upper"].squeeze()[-len(y_test)*2:]
    bb_lo        = df["BB_lower"].squeeze()[-len(y_test)*2:]
    dates_recent = df.index[-len(y_test)*2:]
    ax8.plot(dates_recent, close_recent, color="#1f77b4", linewidth=1,   label="Close")
    ax8.plot(dates_recent, bb_up,        color="#d62728", linewidth=0.8, linestyle="--", label="Upper BB")
    ax8.plot(dates_recent, bb_lo,        color="#2ca02c", linewidth=0.8, linestyle="--", label="Lower BB")
    ax8.fill_between(dates_recent, bb_up, bb_lo, alpha=0.1, color="#1f77b4")
    ax8.set_title("Bollinger Bands (Recent)")
    ax8.set_ylabel("Price (USD)")
    ax8.legend(fontsize=8)
    ax8.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    plt.savefig("crypto_predictor.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Chart saved as crypto_predictor.png")


def predict_next(model, scaler_X, scaler_y, df, feature_cols):
    last_row    = df[feature_cols].iloc[[-1]]
    last_scaled = scaler_X.transform(last_row)
    pred_scaled = model.predict(last_scaled)
    prediction  = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    current     = float(df["Close"].squeeze().iloc[-1])
    change      = prediction - current
    change_pct  = (change / current) * 100

    print(f"\n{'='*45}")
    print(f"  NEXT DAY PREDICTION — {TICKER}")
    print(f"{'='*45}")
    print(f"  Current price:    ${current:,.2f}")
    print(f"  Predicted price:  ${prediction:,.2f}")
    print(f"  Expected change:  {'+' if change >= 0 else ''}{change:,.2f} ({change_pct:+.2f}%)")
    print(f"  Signal:           {'BULLISH' if change > 0 else 'BEARISH'}")
    print(f"{'='*45}\n")
    return prediction


raw_df = fetch_data(TICKER, START_DATE, END_DATE)
df     = add_features(raw_df)

feature_cols = get_feature_cols(df)
X = df[feature_cols].values
y = df["Target"].values

split      = int(len(X) * (1 - TEST_SIZE))
X_train    = X[:split]
X_test     = X[split:]
y_train    = y[:split]
y_test     = y[split:]
test_dates = df.index[split:]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)
y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

print(f"\nTraining samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"Features:         {len(feature_cols)}")

models = {
    "Ridge Regression":  Ridge(alpha=1.0),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=RANDOM_SEED
    ),
}

print(f"\n{'='*50}")
print(f"  MODEL EVALUATION — {TICKER}")
print(f"{'='*50}")

trained    = {}
all_preds  = {}
all_scores = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train_sc)
    preds_sc      = model.predict(X_test_sc)
    preds         = scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).ravel()
    trained[name]    = model
    all_preds[name]  = preds
    all_scores[name] = evaluate(name, y_test, preds)

best_name  = min(all_scores, key=lambda k: all_scores[k]["mape"])
best_model = trained[best_name]
print(f"\n  Best model: {best_name} (MAPE: {all_scores[best_name]['mape']:.2f}%)")

importances = best_model.feature_importances_ if hasattr(best_model, "feature_importances_") \
              else np.abs(best_model.coef_)

plot_results(df, y_test, all_preds, test_dates, feature_cols, importances, all_scores)

predict_next(best_model, scaler_X, scaler_y, df, feature_cols)

print(f"\nTop 10 predictions vs actuals ({best_name}):")
comparison = pd.DataFrame({
    "Date":      test_dates[:10].strftime("%Y-%m-%d"),
    "Actual":    [f"${v:,.2f}" for v in y_test[:10]],
    "Predicted": [f"${v:,.2f}" for v in all_preds[best_name][:10]],
    "Error":     [f"${v:,.2f}" for v in (y_test[:10] - all_preds[best_name][:10])]
})
print(comparison.to_string(index=False))

print("\nTo predict a different coin, change TICKER at the top of the script.")
print("Available examples:", CRYPTO_TICKERS)
print("\nDisclaimer: For educational purposes only. Not financial advice.")
