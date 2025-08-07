"""
Indian Stock Price Prediction System – Main Predictor Class
Author:  Your Name
Updated: 2025-08-05

Key capabilities
----------------
• Data download and cleaning (Yahoo Finance via yfinance)
• Technical-, momentum- and volume-based feature engineering
• Linear Regression, LSTM and Prophet forecasting
• Model evaluation (RMSE, MAE, R², MAPE, Directional Accuracy)
• Future-date prediction and basic trading-signal generation
"""

# --------------------------------------------------------------------------- #
# Standard scientific-python stack
# --------------------------------------------------------------------------- #
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# Sk-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import RobustScaler

# Deep-learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Prophet (optional)
try:
    from prophet import Prophet  # `pip install prophet`
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet  # legacy name
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Global / configurable parameters
# --------------------------------------------------------------------------- #
class Config:
    DATA_DIR = "data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    RESULTS_DIR = "results"
    MODELS_DIR = "models"
    PREDICTIONS_DIR = "predictions"

    MIN_DATA_POINTS = 30
    TEST_SIZE = 0.2

    # LSTM
    LSTM_SEQUENCE_LENGTH = 10
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32

    # Technical indicators
    MA_PERIODS = [5, 10, 20, 50]
    EMA_PERIODS = [12, 26]
    RSI_PERIOD = 14
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2

    # Lag features
    USE_LAG_FEATURES = True
    LAG_PERIODS = [1, 2, 3, 5]

    # Prophet
    PROPHET_SEASONALITY = dict(daily_seasonality=False,
                               weekly_seasonality=True,
                               yearly_seasonality=True)
    PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.05

    # General
    CURRENCY_SYMBOL = "₹"
    VERBOSE = 0


cfg = Config()


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def safe_divide(numerator, denominator, default=0):
    """Element-wise safe divide that suppresses /0 errors."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denominator != 0, numerator / denominator, default)
        return np.where(np.isfinite(out), out, default)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf, clip outliers (0.1 %-99.9 %) and fill NA."""
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.select_dtypes(np.number).columns:
        if df[col].notna().any():
            q_low, q_hi = df[col].quantile([0.001, 0.999])
            df[col] = df[col].clip(q_low, q_hi)

    df = df.fillna(method="ffill").fillna(method="bfill")
    return df


# --------------------------------------------------------------------------- #
# Main predictor
# --------------------------------------------------------------------------- #
class IndianStockPredictor:
    def __init__(self, symbol: str = "RELIANCE.NS", period: str = "2y"):
        self.symbol = symbol.upper()
        self.period = period

        self.data: pd.DataFrame | None = None
        self.processed_data: pd.DataFrame | None = None
        self.scaler = RobustScaler()
        self.models: dict = {}
        self.predictions: dict = {}

        self._create_dirs()

    # ----------------------- directory helpers ---------------------------- #
    def _create_dirs(self):
        for path in (cfg.DATA_DIR, cfg.RAW_DATA_DIR, cfg.PROCESSED_DATA_DIR,
                     cfg.RESULTS_DIR, cfg.MODELS_DIR, cfg.PREDICTIONS_DIR):
            os.makedirs(path, exist_ok=True)

    # ------------------------------ data ---------------------------------- #
    def fetch_data(self):
        """Download OHLCV data via yfinance."""
        try:
            self.data = yf.Ticker(self.symbol).history(period=self.period)
            if self.data.empty:
                raise ValueError(f"No data for symbol {self.symbol}")

            # persist raw
            raw_path = os.path.join(cfg.RAW_DATA_DIR,
                                    f"{self.symbol}_{self.period}_raw.csv")
            self.data.to_csv(raw_path, index=True)
            return self.data
        except Exception as exc:
            print(f"[fetch_data] {exc}")
            return None

    # ----------------------- feature engineering -------------------------- #
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classic MA, EMA, RSI, MACD, BB, etc."""
        min_period = max(cfg.MA_PERIODS + cfg.EMA_PERIODS +
                         [cfg.RSI_PERIOD, cfg.BB_PERIOD])
        if len(df) < min_period:
            # not enough history – only few basic features
            df["MA_5"] = df["Close"].rolling(5, min_periods=1).mean()
            df["Price_Change"] = df["Close"].pct_change().fillna(0)
            df["Volatility"] = df["Price_Change"].rolling(5,
                                                          min_periods=1).std()
            return df

        # SMA
        for p in cfg.MA_PERIODS:
            df[f"MA_{p}"] = df["Close"].rolling(p).mean()

        # EMA
        for p in cfg.EMA_PERIODS:
            df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()

        # MACD
        if all(col in df.columns for col in ["EMA_12", "EMA_26"]):
            df["MACD"] = df["EMA_12"] - df["EMA_26"]
            df["MACD_signal"] = df["MACD"].ewm(span=cfg.MACD_SIGNAL,
                                               adjust=False).mean()
            df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]

        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(cfg.RSI_PERIOD).mean()
        avg_loss = loss.rolling(cfg.RSI_PERIOD).mean()
        rs = safe_divide(avg_gain, avg_loss, 0)
        df["RSI"] = 100 - safe_divide(100, 1 + rs, 50)

        # Bollinger Bands
        ma = df["Close"].rolling(cfg.BB_PERIOD).mean()
        std = df["Close"].rolling(cfg.BB_PERIOD).std()
        df["BB_middle"] = ma
        df["BB_upper"] = ma + cfg.BB_STD * std
        df["BB_lower"] = ma - cfg.BB_STD * std

        # Volume MA for context
        df["Volume_MA"] = df["Volume"].rolling(20).mean()

        # Other quick features
        df["Price_Change"] = df["Close"].pct_change()
        df["Volatility"] = df["Price_Change"].rolling(10).std()
        df["High_Low_Pct"] = safe_divide(df["High"] - df["Low"], df["Close"])

        return df

    def _calculate_fractal_dimension(self, series: pd.Series) -> float:
        """Hurst-like roughness estimate."""
        if len(series) < 10:
            return 0.5
        lags = range(2, min(len(series) // 2, 20))
        tau = [
            np.sqrt(
                np.std(series.values[lag:] - series.values[:-lag])) for lag in lags
        ]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2

    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Micro-structure, volatility and sentiment proxies."""
        df["Price_Efficiency"] = safe_divide(abs(df["Close"] - df["Open"]),
                                             df["High"] - df["Low"] + 1e-8)
        df["Intraday_Return"] = safe_divide(df["Close"] - df["Open"],
                                            df["Open"])
        df["Overnight_Return"] = safe_divide(
            df["Open"] - df["Close"].shift(1), df["Close"].shift(1))

        # Regime / trend strength
        df["Trend_Strength"] = (
            df["Close"].rolling(20).apply(
                lambda s: safe_divide(s.iloc[-1] - s.iloc[0], s.iloc[0])))

        df["Market_Regime"] = np.select(
            [df["Trend_Strength"] > 0.1, df["Trend_Strength"] < -0.1],
            [1, -1],
            default=0)

        # Fractal dimension
        df["Fractal_Dimension"] = df["Close"].rolling(14).apply(
            self._calculate_fractal_dimension)

        # Volume-Price Trend
        df["Volume_Price_Trend"] = df["Volume"] * df["Price_Change"]

        return df

    # ----------------------- prepare full dataset ------------------------- #
    def prepare_data(self):
        if self.data is None:
            self.fetch_data()

        df = self.data.copy()
        df = self.add_technical_indicators(df)
        df = self.add_advanced_features(df)

        # Lag features
        if cfg.USE_LAG_FEATURES:
            for lag in cfg.LAG_PERIODS:
                df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
                df[f"Volume_lag_{lag}"] = df["Volume"].shift(lag)

        # Target
        df["Target"] = df["Close"].shift(-1)
        df = clean_data(df).dropna().iloc[:-1]  # last row Target NaN

        if len(df) < cfg.MIN_DATA_POINTS:
            raise ValueError("Not enough data after preprocessing.")

        self.processed_data = df

        # persist
        path = os.path.join(cfg.PROCESSED_DATA_DIR,
                            f"{self.symbol}_{self.period}_processed.csv")
        df.to_csv(path)

        return df

    # ------------------------ train / test split -------------------------- #
    def split_data(self):
        test_len = max(5, int(len(self.processed_data) * cfg.TEST_SIZE))
        split_point = len(self.processed_data) - test_len
        train = self.processed_data.iloc[:split_point]
        test = self.processed_data.iloc[split_point:]
        return train, test

    # --------------------------- Linear Regression ------------------------ #
    def train_linear_regression(self):
        train, test = self.split_data()

        features = [
            c for c in train.columns if c not in ("Target",) and
            train[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ]

        X_train, y_train = train[features], train["Target"]
        X_test, y_test = test[features], test["Target"]

        lr = LinearRegression().fit(X_train, y_train)

        self.models["linear_regression"] = lr
        self._store_predictions("linear_regression", lr.predict, X_train,
                                X_test, y_train, y_test, test.index)

    # ----------------------------- LSTM ----------------------------------- #
    def prepare_lstm_data(self, seq_len=None):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")

        if seq_len is None:
            seq_len = min(cfg.LSTM_SEQUENCE_LENGTH, len(self.processed_data) //
                          4)
            seq_len = max(5, seq_len)

        series = self.processed_data["Close"].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(series)

        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i - seq_len:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(len(X), seq_len, 1)

        split = int(len(X) * (1 - cfg.TEST_SIZE))
        return (X[:split], X[split:], y[:split], y[split:], seq_len)

    @staticmethod
    def build_lstm_model(seq_len):
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(seq_len, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def train_lstm(self, epochs=None, batch_size=None):
        if not TENSORFLOW_AVAILABLE:
            return

        epochs = epochs or cfg.LSTM_EPOCHS
        batch_size = batch_size or cfg.LSTM_BATCH_SIZE

        # fresh TF session
        K.clear_session()
        try:
            # data
            X_tr, X_te, y_tr, y_te, seq_len = self.prepare_lstm_data()

            model = self.build_lstm_model(seq_len)
            cb = [
                EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(factor=0.2,
                                  patience=5,
                                  min_lr=1e-4,
                                  verbose=0)
            ]
            model.fit(X_tr,
                      y_tr,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(X_te, y_te),
                      verbose=cfg.VERBOSE,
                      callbacks=cb)

            # predictions
            tr_pred = model.predict(X_tr).flatten()
            te_pred = model.predict(X_te).flatten()
            y_tr_inv = self.scaler.inverse_transform(
                y_tr.reshape(-1, 1)).flatten()
            y_te_inv = self.scaler.inverse_transform(
                y_te.reshape(-1, 1)).flatten()
            tr_pred_inv = self.scaler.inverse_transform(
                tr_pred.reshape(-1, 1)).flatten()
            te_pred_inv = self.scaler.inverse_transform(
                te_pred.reshape(-1, 1)).flatten()

            self.models["lstm"] = model
            self.predictions["lstm"] = dict(train_pred=tr_pred_inv,
                                            test_pred=te_pred_inv,
                                            train_actual=y_tr_inv,
                                            test_actual=y_te_inv,
                                            test_dates=self.processed_data.
                                            index[-len(y_te_inv):],
                                            sequence_length=seq_len)
        except Exception as exc:
            print(f"[train_lstm] {exc}")

    # ----------------------------- Prophet ------------------------------- #
    def train_prophet(self):
        if not PROPHET_AVAILABLE:
            return
        try:
            df = pd.DataFrame(
                dict(ds=self.processed_data.index,
                     y=self.processed_data["Close"])).reset_index(drop=True)

            split = int(len(df) * (1 - cfg.TEST_SIZE))
            train_df, test_df = df.iloc[:split], df.iloc[split:]

            m = Prophet(**cfg.PROPHET_SEASONALITY,
                        changepoint_prior_scale=cfg.
                        PROPHET_CHANGEPOINT_PRIOR_SCALE)
            m.fit(train_df)

            future = m.make_future_dataframe(periods=len(test_df))
            fcst = m.predict(future)

            self.models["prophet"] = m
            self.predictions["prophet"] = dict(
                train_pred=fcst["yhat"].iloc[:split].values,
                test_pred=fcst["yhat"].iloc[split:].values,
                train_actual=train_df["y"].values,
                test_actual=test_df["y"].values,
                test_dates=test_df["ds"])
        except Exception as exc:
            print(f"[train_prophet] {exc}")

    # ----------------------- common prediction storage ------------------- #
    def _store_predictions(self, name, predict_fn, X_tr, X_te, y_tr, y_te,
                           test_idx):
        self.predictions[name] = dict(train_pred=predict_fn(X_tr),
                                      test_pred=predict_fn(X_te),
                                      train_actual=y_tr.values,
                                      test_actual=y_te.values,
                                      test_dates=test_idx)

    # ------------------------------ metrics ------------------------------ #
    def evaluate_models(self):
        results = {}
        for name, pred in self.predictions.items():
            rmse = np.sqrt(
                mean_squared_error(pred["test_actual"], pred["test_pred"]))
            mae = mean_absolute_error(pred["test_actual"], pred["test_pred"])
            r2 = r2_score(pred["test_actual"], pred["test_pred"])
            mape = (np.abs(
                safe_divide(pred["test_actual"] - pred["test_pred"],
                            pred["test_actual"]))).mean() * 100

            # direction
            if len(pred["test_actual"]) > 1:
                dir_acc = (np.diff(pred["test_actual"]) > 0 == np.diff(
                    pred["test_pred"]) > 0).mean() * 100
            else:
                dir_acc = 0

            results[name] = dict(RMSE=rmse,
                                 MAE=mae,
                                 R2=r2,
                                 MAPE=mape,
                                 Directional_Accuracy=dir_acc)

        return results

    # -------------------------- future forecast -------------------------- #
    def predict_future(self, days=30):
        if self.processed_data is None:
            self.prepare_data()

        current_price = self.processed_data["Close"].iloc[-1]
        future_dates = [self.processed_data.index[-1] + timedelta(days=i + 1)
                        for i in range(days)]
        future_preds = {}

        # LSTM
        if "lstm" in self.models and "lstm" in self.predictions:
            seq_len = self.predictions["lstm"]["sequence_length"]
            seq = self.scaler.transform(
                self.processed_data["Close"].tail(seq_len).values.reshape(
                    -1, 1))
            preds = []
            for _ in range(days):
                x = seq[-seq_len:].reshape(1, seq_len, 1)
                p = self.models["lstm"].predict(x,
                                                verbose=0)[0, 0]  # scaled
                seq = np.append(seq, p)
                preds.append(p)
            preds = self.scaler.inverse_transform(
                np.array(preds).reshape(-1, 1)).flatten()
            future_preds["LSTM"] = preds

        # Prophet
        if "prophet" in self.models:
            m = self.models["prophet"]
            fcst = m.predict(m.make_future_dataframe(periods=days))
            future_preds["Prophet"] = fcst["yhat"].tail(days).values

        # Linear Regression – simple repeat of last feature row
        if "linear_regression" in self.models:
            lr = self.models["linear_regression"]
            last_row = (self.processed_data.drop("Target", axis=1,
                                                 errors="ignore").iloc[-1:
                                                                       ]).copy()
            preds = []
            for _ in range(days):
                pred = lr.predict(last_row)[0]
                preds.append(pred)
            future_preds["Linear_Regression"] = preds

        # Persist as CSV
        if future_preds:
            pd.DataFrame(future_preds,
                         index=future_dates).to_csv(
                             os.path.join(
                                 cfg.PREDICTIONS_DIR,
                                 f"{self.symbol}_future_predictions.csv"))

        return future_preds, future_dates

    # ----------------------- trading signal toy example ------------------ #
    def generate_trading_signals(self):
        sig = pd.Series(0, index=self.processed_data.index)
        if all(col in self.processed_data.columns
               for col in ("MA_5", "MA_20")):
            cross_up = (self.processed_data["MA_5"] >
                        self.processed_data["MA_20"]) & (
                            self.processed_data["MA_5"].shift(1) <=
                            self.processed_data["MA_20"].shift(1))
            cross_dn = (self.processed_data["MA_5"] <
                        self.processed_data["MA_20"]) & (
                            self.processed_data["MA_5"].shift(1) >=
                            self.processed_data["MA_20"].shift(1))
            sig[cross_up] = 1
            sig[cross_dn] = -1
        return sig

    # --------------------------- convenience ----------------------------- #
    def save_results(self):
        for name, data in self.predictions.items():
            out = pd.DataFrame(dict(Date=data["test_dates"],
                                    Actual=data["test_actual"],
                                    Predicted=data["test_pred"]))
            out.to_csv(
                os.path.join(cfg.RESULTS_DIR,
                             f"{self.symbol}_{name}_predictions.csv"),
                index=False)


# --------------------------------------------------------------------------- #
# Quick utility for REPL / notebook use
# --------------------------------------------------------------------------- #
def quick_predict(symbol="RELIANCE.NS",
                  period="2y",
                  models=("linear_regression", "lstm", "prophet"),
                  future_days=30):
    predictor = IndianStockPredictor(symbol, period)
    predictor.fetch_data()
    predictor.prepare_data()

    if "linear_regression" in models:
        predictor.train_linear_regression()
    if "lstm" in models and TENSORFLOW_AVAILABLE:
        predictor.train_lstm()
    if "prophet" in models and PROPHET_AVAILABLE:
        predictor.train_prophet()

    print("\nModel evaluation:")
    print(predictor.evaluate_models())

    preds, dates = predictor.predict_future(future_days)
    return predictor, preds, dates


if __name__ == "__main__":
    # Example quick run
    p, _, _ = quick_predict("TCS.NS", "2y", ["linear_regression"])
    print("Done.")
