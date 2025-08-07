"""
Configuration file for Indian Stock Price Prediction System
Modify these settings according to your requirements
"""

# Stock Settings
DEFAULT_STOCK = "RELIANCE.NS"  # Default stock to analyze
DEFAULT_PERIOD = "2y"          # Data period: 1y, 2y, 5y, max

# Popular Indian Stocks (NSE)
INDIAN_STOCKS = {
    'RELIANCE.NS': 'Reliance Industries Limited',
    'TCS.NS': 'Tata Consultancy Services Limited',
    'INFY.NS': 'Infosys Limited',
    'HDFCBANK.NS': 'HDFC Bank Limited',
    'ITC.NS': 'ITC Limited',
    'HINDUNILVR.NS': 'Hindustan Unilever Limited',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel Limited',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank Limited',
    'LT.NS': 'Larsen & Toubro Limited',
    'ASIANPAINT.NS': 'Asian Paints Limited',
    'MARUTI.NS': 'Maruti Suzuki India Limited',
    'TITAN.NS': 'Titan Company Limited',
    'WIPRO.NS': 'Wipro Limited',
    'TECHM.NS': 'Tech Mahindra Limited'
}

# Model Parameters
LSTM_EPOCHS = 50              # Number of epochs for LSTM training
LSTM_BATCH_SIZE = 32          # Batch size for LSTM
LSTM_SEQUENCE_LENGTH = 60     # Sequence length for LSTM
LSTM_VALIDATION_SPLIT = 0.2   # Validation split ratio

# Prediction Settings
PREDICTION_DAYS = 30          # Number of days to predict into future
TEST_SIZE = 0.2              # Test set size (20% of data)

# Technical Indicators Parameters
RSI_PERIOD = 14              # RSI period
MACD_FAST = 12               # MACD fast period
MACD_SLOW = 26               # MACD slow period
MACD_SIGNAL = 9              # MACD signal period
BB_PERIOD = 20               # Bollinger Bands period
BB_STD = 2                   # Bollinger Bands standard deviation

# Moving Averages
MA_PERIODS = [5, 10, 20, 50, 200]  # Moving average periods
EMA_PERIODS = [12, 26]             # Exponential moving average periods

# File Paths
DATA_DIR = "data/"
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
MODELS_DIR = "models/saved_models/"
RESULTS_DIR = "results/"
PLOTS_DIR = "results/plots/"
PREDICTIONS_DIR = "results/predictions/"

# Visualization Settings
FIGURE_SIZE = (15, 12)       # Default figure size for plots
DPI = 300                    # Plot resolution
SAVE_PLOTS = True            # Whether to save plots
PLOT_FORMAT = 'png'          # Plot file format

# Model Training Settings
RANDOM_STATE = 42            # Random state for reproducibility
VERBOSE = 1                  # Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)

# Performance Thresholds
MIN_R2_SCORE = 0.7          # Minimum acceptable R² score
MAX_RMSE_THRESHOLD = 100    # Maximum acceptable RMSE
MIN_DATA_POINTS = 100       # Minimum required data points

# Feature Engineering
USE_TECHNICAL_INDICATORS = True  # Whether to use technical indicators
USE_LAG_FEATURES = True         # Whether to create lag features
LAG_PERIODS = [1, 2, 3, 5, 10]  # Lag periods to create

# Prophet Model Settings
PROPHET_SEASONALITY = {
    'daily_seasonality': True,
    'weekly_seasonality': True, 
    'yearly_seasonality': True
}
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.05

# Risk Management
STOP_LOSS_PERCENT = 5.0      # Stop loss percentage
TAKE_PROFIT_PERCENT = 10.0   # Take profit percentage

# Output Settings
SAVE_RESULTS = True          # Whether to save results to files
PRINT_DETAILED_RESULTS = True # Whether to print detailed results
SHOW_PLOTS = True            # Whether to display plots

# Currency Settings
CURRENCY_SYMBOL = "₹"        # Indian Rupee symbol
CURRENCY_FORMAT = "{:.2f}"   # Price formatting

# Data Quality Settings
MIN_VOLUME_THRESHOLD = 1000  # Minimum volume threshold
OUTLIER_THRESHOLD = 3        # Standard deviations for outlier detection
FILL_MISSING_METHOD = 'forward'  # Method to fill missing values

# Alerts and Notifications
ENABLE_ALERTS = False        # Whether to enable trading alerts
ALERT_THRESHOLDS = {
    'high_volatility': 0.05,  # 5% volatility threshold
    'unusual_volume': 2.0,    # 2x average volume
    'price_change': 0.03      # 3% price change
}