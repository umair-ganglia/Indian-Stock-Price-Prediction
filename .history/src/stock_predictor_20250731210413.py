"""
Indian Stock Price Prediction System - Main Predictor Class
Author: [Your Name]
Date: 2025

This module contains the main IndianStockPredictor class that handles:
- Data fetching and preprocessing
- Feature engineering with technical indicators
- Multiple ML model training (Linear Regression, LSTM, Prophet)
- Model evaluation and comparison
- Future price prediction
- Trading signal generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Data fetching
import yfinance as yf

# Machine Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Advanced forecasting
from prophet import Prophet

# Import configuration
import config

class IndianStockPredictor:
    """
    A comprehensive stock price prediction system for Indian stocks
    Combines traditional ML, deep learning, and time-series forecasting
    """
    
    def __init__(self, symbol="RELIANCE.NS", period="2y"):
        """
        Initialize the predictor
        
        Args:
            symbol (str): Stock symbol (e.g., "RELIANCE.NS", "TCS.NS", "INFY.NS")
            period (str): Data period ("1y", "2y", "5y", "max")
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.processed_data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}
        self.predictions = {}
        
        # Create directories if they don't exist
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories for the project"""
        directories = [
            config.DATA_DIR,
            config.RAW_DATA_DIR,
            config.PROCESSED_DATA_DIR,
            config.MODELS_DIR,
            config.RESULTS_DIR,
            config.PLOTS_DIR,
            config.PREDICTIONS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        print(f"📡 Fetching data for {self.symbol}...")
        
        try:
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
                
            print(f"✅ Data fetched successfully: {len(self.data)} records")
            print(f"📅 Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            
            # Save raw data
            self.data.to_csv(f"{config.RAW_DATA_DIR}{self.symbol}_{self.period}_raw.csv")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def add_technical_indicators(self, df):
        """Add technical indicators for better prediction (without TA-Lib)"""
        
        # Moving averages
        for period in config.MA_PERIODS:
            if len(df) >= period:
                df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
        
        # Exponential moving averages
        for period in config.EMA_PERIODS:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # MACD (Moving Average Convergence Divergence)
        if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=config.MACD_SIGNAL).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=config.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=config.RSI_PERIOD).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=config.BB_PERIOD).mean()
        std = df['Close'].rolling(window=config.BB_PERIOD).std()
        df['BB_upper'] = df['BB_middle'] + (std * config.BB_STD)
        df['BB_lower'] = df['BB_middle'] - (std * config.BB_STD)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_rate_of_change'] = df['Volume'].pct_change(periods=10)
        
        # Price features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Range'] = df['High'] - df['Low']
        df['Open_Close_ratio'] = df['Open'] / df['Close']
        df['High_Close_ratio'] = df['High'] / df['Close']
        df['Low_Close_ratio'] = df['Low'] / df['Close']
        
        # Volatility indicators
        df['Volatility'] = df['Price_Change'].rolling(window=10).std()
        df['Volatility_MA'] = df['Volatility'].rolling(window=10).mean()
        
        # Momentum indicators
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['ROC'] = df['Close'].pct_change(periods=10) * 100  # Rate of Change
        
        # Average True Range (ATR)
        df['TR1'] = df['High'] - df['Low']
        df['TR2'] = abs(df['High'] - df['Close'].shift())
        df['TR3'] = abs(df['Low'] - df['Close'].shift())
        df['True_Range'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        df.drop(['TR1', 'TR2', 'TR3'], axis=1, inplace=True)
        
        # Commodity Channel Index (CCI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        tp_ma = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (typical_price - tp_ma) / (0.015 * mad)
        
        return df
    
    def prepare_data(self):
        """Prepare and engineer features for modeling"""
        if self.data is None:
            self.fetch_data()
        
        print("🔧 Preparing data and engineering features...")
        
        # Create a copy for processing
        df = self.data.copy()
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Create lag features if enabled
        if config.USE_LAG_FEATURES:
            for lag in config.LAG_PERIODS:
                df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
                df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
                df[f'High_lag_{lag}'] = df['High'].shift(lag)
                df[f'Low_lag_{lag}'] = df['Low'].shift(lag)
        
        # Create future target (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Data quality checks
        if len(df) < config.MIN_DATA_POINTS:
            raise ValueError(f"Insufficient data points: {len(df)} < {config.MIN_DATA_POINTS}")
        
        self.processed_data = df
        
        # Save processed data
        df.to_csv(f"{config.PROCESSED_DATA_DIR}{self.symbol}_{self.period}_processed.csv")
        
        print(f"✅ Data prepared: {len(df)} records with {len(df.columns)} features")
        
        return df
    
    def split_data(self, test_size=None):
        """Split data into train and test sets"""
        if self.processed_data is None:
            self.prepare_data()
        
        if test_size is None:
            test_size = config.TEST_SIZE
        
        # Calculate split point (time-series split)
        split_point = int(len(self.processed_data) * (1 - test_size))
        
        train_data = self.processed_data[:split_point]
        test_data = self.processed_data[split_point:]
        
        print(f"📊 Data split - Train: {len(train_data)}, Test: {len(test_data)}")
        
        return train_data, test_data
    
    def train_linear_regression(self):
        """Train Linear Regression model"""
        print("🤖 Training Linear Regression model...")
        
        train_data, test_data = self.split_data()
        
        # Select features (excluding target and non-numeric columns)
        feature_columns = [col for col in train_data.columns 
                          if col not in ['Target'] and 
                          train_data[col].dtype in ['int64', 'float64']]
        
        X_train = train_data[feature_columns]
        y_train = train_data['Target']
        X_test = test_data[feature_columns]
        y_test = test_data['Target']
        
        # Handle any remaining NaN values
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())  # Use train mean for test
        
        # Train model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = lr_model.predict(X_train)
        test_pred = lr_model.predict(X_test)
        
        # Store model and predictions
        self.models['linear_regression'] = lr_model
        self.predictions['linear_regression'] = {
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_actual': y_train.values,
            'test_actual': y_test.values,
            'test_dates': test_data.index,
            'feature_importance': dict(zip(feature_columns, lr_model.coef_))
        }
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"📈 Linear Regression - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}")
        
        return lr_model
    
    def prepare_lstm_data(self, sequence_length=None):
        """Prepare data for LSTM model"""
        if self.processed_data is None:
            self.prepare_data()
        
        if sequence_length is None:
            sequence_length = config.LSTM_SEQUENCE_LENGTH
        
        # Use only Close price for LSTM (simpler and more stable)
        data = self.processed_data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split data
        split_point = int(len(X) * (1 - config.TEST_SIZE))
        
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        return X_train, X_test, y_train, y_test
    
    def build_lstm_model(self, sequence_length=None):
        """Build LSTM model architecture"""
        if sequence_length is None:
            sequence_length = config.LSTM_SEQUENCE_LENGTH
            
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def train_lstm(self, sequence_length=None, epochs=None, batch_size=None):
        """Train LSTM model"""
        print("🧠 Training LSTM model...")
        
        if sequence_length is None:
            sequence_length = config.LSTM_SEQUENCE_LENGTH
        if epochs is None:
            epochs = config.LSTM_EPOCHS
        if batch_size is None:
            batch_size = config.LSTM_BATCH_SIZE
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_lstm_data(sequence_length)
        
        # Build model
        lstm_model = self.build_lstm_model(sequence_length)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True,
            verbose=0
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=0.0001,
            verbose=0
        )
        
        # Train model
        history = lstm_model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=config.VERBOSE
        )
        
        # Make predictions
        train_pred = lstm_model.predict(X_train, verbose=0)
        test_pred = lstm_model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        train_pred = self.scaler.inverse_transform(train_pred)
        test_pred = self.scaler.inverse_transform(test_pred)
        y_train_actual = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Store model and predictions
        self.models['lstm'] = lstm_model
        self.predictions['lstm'] = {
            'train_pred': train_pred.flatten(),
            'test_pred': test_pred.flatten(),
            'train_actual': y_train_actual.flatten(),
            'test_actual': y_test_actual.flatten(),
            'history': history,
            'test_dates': self.processed_data.index[-len(y_test):]
        }
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
        test_r2 = r2_score(y_test_actual, test_pred)
        
        print(f"🧠 LSTM - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}")
        
        return lstm_model
    
    def train_prophet(self):
        """Train Prophet model for time-series forecasting"""
        print("📊 Training Prophet model...")
        
        if self.processed_data is None:
            self.prepare_data()
        
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': self.processed_data.index,
            'y': self.processed_data['Close']
        })
        
        # Split data
        split_point = int(len(prophet_data) * (1 - config.TEST_SIZE))
        train_data = prophet_data[:split_point]
        test_data = prophet_data[split_point:]
        
        # Train Prophet model
        prophet_model = Prophet(
            daily_seasonality=config.PROPHET_SEASONALITY['daily_seasonality'],
            weekly_seasonality=config.PROPHET_SEASONALITY['weekly_seasonality'],
            yearly_seasonality=config.PROPHET_SEASONALITY['yearly_seasonality'],
            changepoint_prior_scale=config.PROPHET_CHANGEPOINT_PRIOR_SCALE
        )
        
        prophet_model.fit(train_data)
        
        # Make predictions
        future = prophet_model.make_future_dataframe(periods=len(test_data))
        forecast = prophet_model.predict(future)
        
        # Extract predictions
        train_pred = forecast['yhat'][:len(train_data)].values
        test_pred = forecast['yhat'][len(train_data):].values
        
        # Store model and predictions
        self.models['prophet'] = prophet_model
        self.predictions['prophet'] = {
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_actual': train_data['y'].values,
            'test_actual': test_data['y'].values,
            'forecast': forecast,
            'test_dates': test_data['ds']
        }
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(train_data['y'], train_pred))
        test_rmse = np.sqrt(mean_squared_error(test_data['y'], test_pred))
        test_r2 = r2_score(test_data['y'], test_pred)
        
        print(f"📊 Prophet - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}")
        
        return prophet_model
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION SUMMARY")
        print("="*60)
        
        results = {}
        
        for model_name, pred_data in self.predictions.items():
            test_rmse = np.sqrt(mean_squared_error(pred_data['test_actual'], pred_data['test_pred']))
            test_mae = mean_absolute_error(pred_data['test_actual'], pred_data['test_pred'])
            test_r2 = r2_score(pred_data['test_actual'], pred_data['test_pred'])
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((pred_data['test_actual'] - pred_data['test_pred']) / pred_data['test_actual'])) * 100
            
            # Calculate directional accuracy
            actual_direction = np.diff(pred_data['test_actual']) > 0
            pred_direction = np.diff(pred_data['test_pred']) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            results[model_name] = {
                'RMSE': test_rmse,
                'MAE': test_mae,
                'R²': test_r2,
                'MAPE': mape,
                'Directional_Accuracy': directional_accuracy
            }
            
            print(f"\n🤖 {model_name.upper()}:")
            print(f"   RMSE: {config.CURRENCY_SYMBOL}{test_rmse:.2f}")
            print(f"   MAE:  {config.CURRENCY_SYMBOL}{test_mae:.2f}")
            print(f"   R²:   {test_r2:.4f}")
            print(f"   MAPE: {mape:.2f}%")
            print(f"   Direction Accuracy: {directional_accuracy:.1f}%")
        
        # Save results
        results_df = pd.DataFrame(results).T
        results_df.to_csv(f"{config.RESULTS_DIR}model_comparison_{self.symbol}.csv")
        
        return results
    
    def plot_predictions(self, figsize=None):
        """Plot predictions from all models"""
        if figsize is None:
            figsize = config.FIGURE_SIZE
            
        n_models = len(self.predictions)
        fig, axes = plt.subplots(n_models, 1, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        plt.style.use('seaborn-v0_8')
        
        for i, (model_name, pred_data) in enumerate(self.predictions.items()):
            ax = axes[i]
            
            # Plot actual vs predicted
            test_dates = pred_data['test_dates']
            ax.plot(test_dates, pred_data['test_actual'], 
                   label='Actual', color='#2E86AB', linewidth=2, alpha=0.8)
            ax.plot(test_dates, pred_data['test_pred'], 
                   label='Predicted', color='#F24236', linewidth=2, alpha=0.8)
            
            ax.set_title(f'{model_name.upper()} - {self.symbol} Stock Price Prediction', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel(f'Price ({config.CURRENCY_SYMBOL})', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Calculate and display metrics
            rmse = np.sqrt(mean_squared_error(pred_data['test_actual'], pred_data['test_pred']))
            r2 = r2_score(pred_data['test_actual'], pred_data['test_pred'])
            mape = np.mean(np.abs((pred_data['test_actual'] - pred_data['test_pred']) / pred_data['test_actual'])) * 100
            
            metrics_text = f'RMSE: {config.CURRENCY_SYMBOL}{rmse:.2f}\nR²: {r2:.4f}\nMAPE: {mape:.1f}%'
            ax.text(0.02, 0.98, metrics_text, 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   fontsize=10)
        
        plt.tight_layout()
        
        if config.SAVE_PLOTS:
            plt.savefig(f"{config.PLOTS_DIR}{self.symbol}_predictions.{config.PLOT_FORMAT}", 
                       dpi=config.DPI, bbox_inches='tight')
        
        if config.SHOW_PLOTS:
            plt.show()
    
    def predict_future(self, days=None):
        """Predict future stock prices"""
        if days is None:
            days = config.PREDICTION_DAYS
            
        print(f"\n🔮 Predicting next {days} days...")
        
        future_predictions = {}
        
        # LSTM predictions
        if 'lstm' in self.models:
            last_sequence = self.scaler.transform(
                self.processed_data['Close'].tail(config.LSTM_SEQUENCE_LENGTH).values.reshape(-1, 1)
            )
            
            future_pred = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days):
                # Reshape for LSTM input
                input_seq = current_sequence[-config.LSTM_SEQUENCE_LENGTH:].reshape(1, config.LSTM_SEQUENCE_LENGTH, 1)
                next_pred = self.models['lstm'].predict(input_seq, verbose=0)
                future_pred.append(next_pred[0, 0])
                
                # Update sequence
                current_sequence = np.append(current_sequence, next_pred)
            
            # Inverse transform
            future_pred = self.scaler.inverse_transform(np.array(future_pred).reshape(-1, 1))
            future_predictions['LSTM'] = future_pred.flatten()
        
        # Prophet predictions
        if 'prophet' in self.models:
            future_dates = self.models['prophet'].make_future_dataframe(periods=days)
            forecast = self.models['prophet'].predict(future_dates)
            future_predictions['Prophet'] = forecast['yhat'].tail(days).values
        
        # Linear Regression predictions (simplified approach)
        if 'linear_regression' in self.models:
            # Use last known values to predict
            last_features = self.processed_data.iloc[-1:].drop(['Target'], axis=1)
            last_features = last_features.select_dtypes(include=[np.number]).fillna(last_features.mean())
            
            lr_predictions = []
            for _ in range(days):
                next_pred = self.models['linear_regression'].predict(last_features)[0]
                lr_predictions.append(next_pred)
                # Simple approach: assume features remain similar
            
            future_predictions['Linear_Regression'] = lr_predictions
        
        # Create future dates
        last_date = self.processed_data.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame(future_predictions, index=future_dates)
        predictions_df.to_csv(f"{config.PREDICTIONS_DIR}{self.symbol}_future_predictions.csv")
        
        # Display predictions
        print(f"\n📈 Future Price Predictions for {self.symbol}:")
        print("-" * 50)
        
        current_price = self.processed_data['Close'].iloc[-1]
        print(f"Current Price: {config.CURRENCY_SYMBOL}{current_price:.2f}")
        print()
        
        for model_name, predictions in future_predictions.items():
            print(f"{model_name}:")
            for i, (date, price) in enumerate(zip(future_dates[:10], predictions[:10])):
                change = ((price - current_price) / current_price) * 100
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"  {date.strftime('%Y-%m-%d')}: {config.CURRENCY_SYMBOL}{price:.2f} ({change:+.1f}%) {direction}")
            
            if days > 10:
                print(f"  ... (showing first 10 of {days} predictions)")
            print()
        
        return future_predictions, future_dates
    
    def generate_trading_signals(self):
        """Generate buy/sell signals based on technical indicators"""
        if self.processed_data is None:
            self.prepare_data()
        
        df = self.processed_data.copy()
        signals = pd.Series(0, index=df.index)  # 0 = Hold, 1 = Buy, -1 = Sell
        
        # Moving average crossover strategy
        if 'MA_5' in df.columns and 'MA_20' in df.columns:
            signals[(df['MA_5'] > df['MA_20']) & (df['MA_5'].shift(1) <= df['MA_20'].shift(1))] = 1  # Golden cross
            signals[(df['MA_5'] < df['MA_20']) & (df['MA_5'].shift(1) >= df['MA_20'].shift(1))] = -1  # Death cross
        
        # RSI-based signals
        if 'RSI' in df.columns:
            signals[(df['RSI'] < 30) & (signals != -1)] = 1  # Oversold - Buy
            signals[(df['RSI'] > 70) & (signals != 1)] = -1  # Overbought - Sell
        
        # MACD signals
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            signals[(df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))] = 1
            signals[(df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))] = -1
        
        # Bollinger Bands signals
        if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'Close']):
            signals[(df['Close'] < df['BB_lower']) & (signals != -1)] = 1  # Price below lower band - Buy
            signals[(df['Close'] > df['BB_upper']) & (signals != 1)] = -1  # Price above upper band - Sell
        
        return signals
    
    def save_results(self):
        """Save all results to files"""
        print("💾 Saving results...")
        
        # Save processed data
        if self.processed_data is not None:
            self.processed_data.to_csv(f"{config.RESULTS_DIR}{self.symbol}_processed_data.csv")
        
        # Save model predictions
        for model_name, pred_data in self.predictions.items():
            results = pd.DataFrame({
                'Date': pred_data['test_dates'],
                'Actual': pred_data['test_actual'],
                'Predicted': pred_data['test_pred']
            })
            results.to_csv(f"{config.RESULTS_DIR}{self.symbol}_{model_name}_predictions.csv", index=False)
        
        # Save trading signals
        signals = self.generate_trading_signals()
        signals_df = pd.DataFrame({
            'Date': signals.index,
            'Signal': signals.values
        })
        signals_df.to_csv(f"{config.RESULTS_DIR}{self.symbol}_trading_signals.csv", index=False)
        
        print("✅ Results saved successfully!")
    
    def get_stock_info(self):
        """Get additional stock information"""
        try:
            stock = yf.Ticker(self.symbol)
            info = stock.info
            
            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'beta': info.get('beta', 'N/A')
            }
        except:
            return None
    
    def plot_technical_indicators(self):
        """Plot technical indicators"""
        if self.processed_data is None:
            self.prepare_data()
        
        df = self.processed_data.tail(200).copy()  # Last 200 days
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # Price and Moving Averages
        ax1 = axes[0]
        ax1.plot(df.index, df['Close'], label='Close Price', linewidth=2)
        if 'MA_20' in df.columns:
            ax1.plot(df.index, df['MA_20'], label='MA 20', alpha=0.7)
        if 'MA_50' in df.columns:
            ax1.plot(df.index, df['MA_50'], label='MA 50', alpha=0.7)
        ax1.set_title(f'{self.symbol} - Price and Moving Averages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI
        ax2 = axes[1]
        if 'RSI' in df.columns:
            ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax2.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
        ax2.set_title('RSI (Relative Strength Index)')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # MACD
        ax3 = axes[2]
        if all(col in df.columns for col in ['MACD', 'MACD_signal']):
            ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
            ax3.plot(df.index, df['MACD_signal'], label='Signal', color='red')
            if 'MACD_histogram' in df.columns:
                ax3.bar(df.index, df['MACD_histogram'], label='Histogram', alpha=0.3, color='gray')
        ax3.set_title('MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Volume
        ax4 = axes[3]
        ax4.bar(df.index, df['Volume'], alpha=0.7, color='lightblue')
        if 'Volume_MA' in df.columns:
            ax4.plot(df.index, df['Volume_MA'], color='red', label='Volume MA')
        ax4.set_title('Trading Volume')
        ax4.set_ylabel('Volume')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if config.SAVE_PLOTS:
            plt.savefig(f"{config.PLOTS_DIR}{self.symbol}_technical_indicators.{config.PLOT_FORMAT}", 
                       dpi=config.DPI, bbox_inches='tight')
        
        if config.SHOW_PLOTS:
            plt.show()


# Helper function for easy usage
def quick_predict(symbol, period="2y", models=['linear_regression', 'lstm', 'prophet']):
    """Quick prediction function for easy usage"""
    predictor = IndianStockPredictor(symbol=symbol, period=period)
    
    # Fetch and prepare data
    predictor.fetch_data()
    predictor.prepare_data()
    
    # Train selected models
    if 'linear_regression' in models:
        predictor.train_linear_regression()
    if 'lstm' in models:
        predictor.train_lstm()
    if 'prophet' in models:
        predictor.train_prophet()
    
    # Evaluate and predict
    predictor.evaluate_models()
    future_pred, future_dates = predictor.predict_future()
    
    return predictor, future_pred