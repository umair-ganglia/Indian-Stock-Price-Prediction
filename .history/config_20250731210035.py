# Configuration settings for the Stock Price Prediction System

# Data paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"

# Model paths
MODEL_DIR = "models/saved_models"

# Results paths
PLOTS_DIR = "results/plots"
PREDICTIONS_DIR = "results/predictions"

# Model parameters
SEQUENCE_LENGTH = 60  # Number of time steps to look back
TRAIN_SPLIT = 0.8    # Proportion of data to use for training
VALIDATION_SPLIT = 0.1  # Proportion of data to use for validation
TEST_SPLIT = 0.1     # Proportion of data to use for testing

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Stock data parameters
TICKER_SYMBOL = "AAPL"  # Default stock symbol
START_DATE = "2010-01-01"
END_DATE = "2025-07-31"  # Current date
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']  # Features to use for prediction
