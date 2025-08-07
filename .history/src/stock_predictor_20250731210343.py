"""
Stock Predictor Module for Indian Stock Market
Implements various prediction models and analysis tools
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import config

class IndianStockPredictor:
    def __init__(self, symbol=config.DEFAULT_STOCK, period=config.DEFAULT_PERIOD):
        """
        Initialize the stock predictor
        
        Args:
            symbol (str): Stock symbol (NSE)
            period (str): Data period to fetch
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.scaled_data = None
        self.scaler = MinMaxScaler()
        self.models = {}
        self.predictions = {}
        self.evaluation = {}
        
        # Create required directories
        for directory in [config.MODELS_DIR, config.PLOTS_DIR, config.PREDICTIONS_DIR]:
            os.makedirs(directory, exist_ok=True)

    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            # Save raw data
            raw_file_path = os.path.join(config.RAW_DATA_DIR, f"{self.symbol}_raw.csv")
            self.data.to_csv(raw_file_path)
            
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None

    def prepare_data(self):
        """Prepare data for modeling"""
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")

        # Calculate technical indicators
        if config.USE_TECHNICAL_INDICATORS:
            self._add_technical_indicators()

        # Create lag features
        if config.USE_LAG_FEATURES:
            self._add_lag_features()

        # Scale the features
        self.scaled_data = self.scaler.fit_transform(self.data[['Close']])
        
        # Save processed data
        processed_file_path = os.path.join(config.PROCESSED_DATA_DIR, f"{self.symbol}_processed.csv")
        self.data.to_csv(processed_file_path)

    def _add_technical_indicators(self):
        """Add technical indicators to the dataset"""
        # Moving averages
        for period in config.MA_PERIODS:
            self.data[f'MA_{period}'] = self.data['Close'].rolling(window=period).mean()
            
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=config.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=config.RSI_PERIOD).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        self.data['BB_middle'] = self.data['Close'].rolling(window=config.BB_PERIOD).mean()
        std = self.data['Close'].rolling(window=config.BB_PERIOD).std()
        self.data['BB_upper'] = self.data['BB_middle'] + (std * config.BB_STD)
        self.data['BB_lower'] = self.data['BB_middle'] - (std * config.BB_STD)

    def _add_lag_features(self):
        """Add lagged price features"""
        for lag in config.LAG_PERIODS:
            self.data[f'Price_Lag_{lag}'] = self.data['Close'].shift(lag)

    def _create_sequences(self, data, seq_length):
        """Create sequences for LSTM"""
        sequences = []
        targets = []
        
        for i in range(len(data) - seq_length):
            sequences.append(data[i:(i + seq_length)])
            targets.append(data[i + seq_length])
            
        return np.array(sequences), np.array(targets)

    def train_linear_regression(self):
        """Train Linear Regression model"""
        # Prepare data
        X = np.arange(len(self.scaled_data)).reshape(-1, 1)
        y = self.scaled_data
        
        # Split data
        train_size = int(len(X) * (1 - config.TEST_SIZE))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Make predictions
        lr_pred = lr_model.predict(X_test)
        
        # Store results
        self.models['linear'] = lr_model
        self.predictions['linear'] = self.scaler.inverse_transform(lr_pred.reshape(-1, 1))
        
        # Evaluate
        self.evaluation['linear'] = {
            'RMSE': np.sqrt(mean_squared_error(
                self.scaler.inverse_transform(y_test),
                self.predictions['linear']
            )),
            'R2': r2_score(y_test, lr_pred)
        }

    def train_lstm(self, epochs=config.LSTM_EPOCHS):
        """Train LSTM model"""
        # Create sequences
        X, y = self._create_sequences(self.scaled_data, config.LSTM_SEQUENCE_LENGTH)
        
        # Split data
        train_size = int(len(X) * (1 - config.TEST_SIZE))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(config.LSTM_SEQUENCE_LENGTH, 1)),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=config.LSTM_BATCH_SIZE,
            validation_split=config.LSTM_VALIDATION_SPLIT,
            verbose=0
        )
        
        # Make predictions
        lstm_pred = model.predict(X_test)
        
        # Store results
        self.models['lstm'] = model
        self.predictions['lstm'] = self.scaler.inverse_transform(lstm_pred)
        
        # Evaluate
        self.evaluation['lstm'] = {
            'RMSE': np.sqrt(mean_squared_error(
                self.scaler.inverse_transform(y_test),
                self.predictions['lstm']
            )),
            'R2': r2_score(y_test, lstm_pred)
        }

    def train_prophet(self):
        """Train Prophet model"""
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': self.data.index,
            'y': self.data['Close']
        })
        
        # Create and train model
        model = Prophet(**config.PROPHET_SEASONALITY)
        model.fit(prophet_data)
        
        # Make predictions
        future_dates = model.make_future_dataframe(periods=len(self.data) * config.TEST_SIZE)
        forecast = model.predict(future_dates)
        
        # Store results
        self.models['prophet'] = model
        self.predictions['prophet'] = forecast.tail(int(len(self.data) * config.TEST_SIZE))['yhat'].values
        
        # Evaluate
        actual = self.data['Close'].iloc[-len(self.predictions['prophet']):]
        self.evaluation['prophet'] = {
            'RMSE': np.sqrt(mean_squared_error(actual, self.predictions['prophet'])),
            'R2': r2_score(actual, self.predictions['prophet'])
        }

    def evaluate_models(self):
        """Return evaluation metrics for all models"""
        print("\nModel Performance Metrics:")
        print("-" * 40)
        
        for model in self.evaluation:
            print(f"\n{model.upper()}:")
            print(f"RMSE: {self.evaluation[model]['RMSE']:.2f}")
            print(f"R² Score: {self.evaluation[model]['R2']:.4f}")
        
        return self.evaluation

    def predict_future(self, days=config.PREDICTION_DAYS):
        """Generate future predictions"""
        future_dates = pd.date_range(
            start=self.data.index[-1],
            periods=days + 1,
            freq='B'
        )[1:]
        
        predictions = {}
        
        # Prophet predictions
        if 'prophet' in self.models:
            future = self.models['prophet'].make_future_dataframe(periods=days)
            forecast = self.models['prophet'].predict(future)
            predictions['prophet'] = forecast.tail(days)['yhat'].values
        
        # LSTM predictions (recursive)
        if 'lstm' in self.models:
            last_sequence = self.scaled_data[-config.LSTM_SEQUENCE_LENGTH:]
            lstm_pred = []
            
            for _ in range(days):
                next_pred = self.models['lstm'].predict(
                    last_sequence.reshape(1, config.LSTM_SEQUENCE_LENGTH, 1)
                )
                lstm_pred.append(next_pred[0, 0])
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = next_pred
                
            predictions['lstm'] = self.scaler.inverse_transform(
                np.array(lstm_pred).reshape(-1, 1)
            ).flatten()
        
        # Linear regression predictions
        if 'linear' in self.models:
            future_indices = np.arange(
                len(self.scaled_data),
                len(self.scaled_data) + days
            ).reshape(-1, 1)
            
            predictions['linear'] = self.scaler.inverse_transform(
                self.models['linear'].predict(future_indices).reshape(-1, 1)
            ).flatten()
        
        return predictions, future_dates

    def generate_trading_signals(self):
        """Generate trading signals based on technical indicators"""
        signals = pd.Series(index=self.data.index, data=0)
        
        # Generate signals based on RSI
        signals[self.data['RSI'] < 30] = 1    # Oversold - Buy signal
        signals[self.data['RSI'] > 70] = -1   # Overbought - Sell signal
        
        # Generate signals based on Bollinger Bands
        signals[self.data['Close'] < self.data['BB_lower']] = 1     # Price below lower band - Buy signal
        signals[self.data['Close'] > self.data['BB_upper']] = -1    # Price above upper band - Sell signal
        
        return signals

    def plot_predictions(self):
        """Plot actual vs predicted values"""
        plt.figure(figsize=config.FIGURE_SIZE)
        
        # Plot actual prices
        plt.plot(self.data.index[-100:], self.data['Close'].tail(100),
                label='Actual', color='black', alpha=0.7)
        
        # Plot predictions for each model
        colors = {'linear': 'blue', 'lstm': 'green', 'prophet': 'red'}
        for model, predictions in self.predictions.items():
            plt.plot(self.data.index[-len(predictions):],
                    predictions, label=f'{model.upper()} Predictions',
                    color=colors[model], alpha=0.7)
        
        plt.title(f'{self.symbol} Stock Price Predictions')
        plt.xlabel('Date')
        plt.ylabel(f'Price ({config.CURRENCY_SYMBOL})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        if config.SAVE_PLOTS:
            plt.savefig(
                os.path.join(config.PLOTS_DIR, f"{self.symbol}_predictions.{config.PLOT_FORMAT}"),
                dpi=config.DPI
            )
        
        if config.SHOW_PLOTS:
            plt.show()
        plt.close()

    def save_results(self):
        """Save predictions and evaluation metrics"""
        # Save predictions
        for model, preds in self.predictions.items():
            pd.DataFrame({
                'Date': self.data.index[-len(preds):],
                'Actual': self.data['Close'].tail(len(preds)),
                'Predicted': preds
            }).to_csv(os.path.join(
                config.PREDICTIONS_DIR,
                f"{self.symbol}_{model}_predictions.csv"
            ))
        
        # Save evaluation metrics
        pd.DataFrame(self.evaluation).to_csv(
            os.path.join(config.RESULTS_DIR, f"{self.symbol}_evaluation.csv")
        )
