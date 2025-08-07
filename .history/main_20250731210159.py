"""
Main execution file for the Indian Stock Price Prediction System
"""

import os
import sys
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt

# Import configuration
import config as cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        cfg.DATA_DIR,
        cfg.RAW_DATA_DIR,
        cfg.PROCESSED_DATA_DIR,
        cfg.MODELS_DIR,
        cfg.RESULTS_DIR,
        cfg.PLOTS_DIR,
        cfg.PREDICTIONS_DIR
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

def fetch_stock_data(symbol=cfg.DEFAULT_STOCK, period=cfg.DEFAULT_PERIOD):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        symbol (str): Stock symbol
        period (str): Data period (e.g., '2y' for 2 years)
    
    Returns:
        pd.DataFrame: Historical stock data
    """
    try:
        logging.info(f"Fetching data for {symbol}")
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        
        # Save raw data
        raw_file_path = os.path.join(cfg.RAW_DATA_DIR, f"{symbol}_raw.csv")
        df.to_csv(raw_file_path)
        logging.info(f"Raw data saved to {raw_file_path}")
        
        return df
    
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def validate_data(df):
    """
    Validate and clean the data
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if df is None or df.empty:
        logging.error("No data to validate")
        return None
    
    try:
        # Check for minimum data points
        if len(df) < cfg.MIN_DATA_POINTS:
            logging.warning(f"Insufficient data points: {len(df)} < {cfg.MIN_DATA_POINTS}")
            return None
        
        # Handle missing values
        df = df.fillna(method=cfg.FILL_MISSING_METHOD)
        
        # Remove rows with volume below threshold
        df = df[df['Volume'] >= cfg.MIN_VOLUME_THRESHOLD]
        
        # Handle outliers
        for column in ['Open', 'High', 'Low', 'Close']:
            mean = df[column].mean()
            std = df[column].std()
            df = df[abs(df[column] - mean) <= cfg.OUTLIER_THRESHOLD * std]
        
        return df
    
    except Exception as e:
        logging.error(f"Error in data validation: {str(e)}")
        return None

def main():
    """Main execution function"""
    try:
        # Setup directories
        setup_directories()
        
        # Get list of stocks to analyze
        stocks = list(cfg.INDIAN_STOCKS.keys())
        
        for symbol in stocks:
            logging.info(f"\nProcessing {symbol} ({cfg.INDIAN_STOCKS[symbol]})")
            
            # Fetch data
            df = fetch_stock_data(symbol)
            if df is None:
                continue
            
            # Validate and clean data
            df = validate_data(df)
            if df is None:
                continue
            
            # Save processed data
            processed_file_path = os.path.join(cfg.PROCESSED_DATA_DIR, f"{symbol}_processed.csv")
            df.to_csv(processed_file_path)
            logging.info(f"Processed data saved to {processed_file_path}")
            
            # Basic visualization
            if cfg.SHOW_PLOTS:
                plt.figure(figsize=cfg.FIGURE_SIZE)
                plt.plot(df.index, df['Close'], label='Close Price')
                plt.title(f'{cfg.INDIAN_STOCKS[symbol]} Stock Price')
                plt.xlabel('Date')
                plt.ylabel(f'Price ({cfg.CURRENCY_SYMBOL})')
                plt.legend()
                
                if cfg.SAVE_PLOTS:
                    plot_path = os.path.join(cfg.PLOTS_DIR, f"{symbol}_price.{cfg.PLOT_FORMAT}")
                    plt.savefig(plot_path, dpi=cfg.DPI)
                    logging.info(f"Plot saved to {plot_path}")
                
                if cfg.SHOW_PLOTS:
                    plt.show()
                plt.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nProgram terminated by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    finally:
        logging.info("Program finished")
