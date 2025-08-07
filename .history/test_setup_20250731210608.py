"""
Test suite for the Indian Stock Price Prediction System
"""

import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src directory to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stock_predictor import IndianStockPredictor
import config

class TestStockPredictor(unittest.TestCase):
    """Test cases for the IndianStockPredictor class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("\n🔍 Setting up test environment...")
        
        # Create test directories if they don't exist
        for directory in [
            config.DATA_DIR,
            config.RAW_DATA_DIR,
            config.PROCESSED_DATA_DIR,
            config.MODELS_DIR,
            config.RESULTS_DIR,
            config.PLOTS_DIR,
            config.PREDICTIONS_DIR
        ]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize predictor with a test stock
        cls.predictor = IndianStockPredictor(
            symbol="TCS.NS",  # Using TCS as test stock
            period="6mo"      # Using shorter period for testing
        )
    
    def setUp(self):
        """Set up each test"""
        print(f"\n📋 Running: {self._testMethodName}")
    
    def test_01_initialization(self):
        """Test predictor initialization"""
        self.assertIsNotNone(self.predictor)
        self.assertEqual(self.predictor.symbol, "TCS.NS")
        self.assertEqual(self.predictor.period, "6mo")
        self.assertIsNone(self.predictor.data)
    
    def test_02_data_fetching(self):
        """Test data fetching functionality"""
        data = self.predictor.fetch_data()
        
        # Check if data was fetched
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for column in required_columns:
            self.assertIn(column, data.columns)
        
        # Check data quality
        self.assertGreater(len(data), 0)
        self.assertFalse(data['Close'].isnull().any())
    
    def test_03_data_preparation(self):
        """Test data preparation and feature engineering"""
        self.predictor.prepare_data()
        
        # Check if technical indicators were added
        if config.USE_TECHNICAL_INDICATORS:
            self.assertIn('RSI', self.predictor.data.columns)
            self.assertIn('BB_upper', self.predictor.data.columns)
            self.assertIn('BB_lower', self.predictor.data.columns)
            
            # Check moving averages
            for period in config.MA_PERIODS:
                self.assertIn(f'MA_{period}', self.predictor.data.columns)
        
        # Check lag features
        if config.USE_LAG_FEATURES:
            for lag in config.LAG_PERIODS:
                self.assertIn(f'Price_Lag_{lag}', self.predictor.data.columns)
        
        # Check scaled data
        self.assertIsNotNone(self.predictor.scaled_data)
    
    def test_04_linear_regression(self):
        """Test linear regression model"""
        self.predictor.train_linear_regression()
        
        # Check if model was created
        self.assertIn('linear', self.predictor.models)
        self.assertIn('linear', self.predictor.predictions)
        self.assertIn('linear', self.predictor.evaluation)
        
        # Check evaluation metrics
        self.assertIn('RMSE', self.predictor.evaluation['linear'])
        self.assertIn('R2', self.predictor.evaluation['linear'])
        
        # Verify predictions
        self.assertGreater(len(self.predictor.predictions['linear']), 0)
    
    def test_05_lstm_model(self):
        """Test LSTM model"""
        self.predictor.train_lstm(epochs=2)  # Using fewer epochs for testing
        
        # Check if model was created
        self.assertIn('lstm', self.predictor.models)
        self.assertIn('lstm', self.predictor.predictions)
        self.assertIn('lstm', self.predictor.evaluation)
        
        # Check evaluation metrics
        self.assertIn('RMSE', self.predictor.evaluation['lstm'])
        self.assertIn('R2', self.predictor.evaluation['lstm'])
        
        # Verify predictions
        self.assertGreater(len(self.predictor.predictions['lstm']), 0)
    
    def test_06_prophet_model(self):
        """Test Prophet model"""
        self.predictor.train_prophet()
        
        # Check if model was created
        self.assertIn('prophet', self.predictor.models)
        self.assertIn('prophet', self.predictor.predictions)
        self.assertIn('prophet', self.predictor.evaluation)
        
        # Check evaluation metrics
        self.assertIn('RMSE', self.predictor.evaluation['prophet'])
        self.assertIn('R2', self.predictor.evaluation['prophet'])
        
        # Verify predictions
        self.assertGreater(len(self.predictor.predictions['prophet']), 0)
    
    def test_07_future_predictions(self):
        """Test future predictions"""
        predictions, future_dates = self.predictor.predict_future(days=5)
        
        # Check predictions format
        self.assertIsInstance(predictions, dict)
        self.assertIsInstance(future_dates, pd.DatetimeIndex)
        
        # Check prediction length
        self.assertEqual(len(future_dates), 5)
        
        # Check if we have predictions from each model
        for model in ['linear', 'lstm', 'prophet']:
            self.assertIn(model, predictions)
            self.assertEqual(len(predictions[model]), 5)
    
    def test_08_trading_signals(self):
        """Test trading signals generation"""
        signals = self.predictor.generate_trading_signals()
        
        # Check signals format
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.predictor.data))
        
        # Check signal values
        self.assertTrue(all(s in [-1, 0, 1] for s in signals.unique()))
    
    def test_09_save_results(self):
        """Test results saving functionality"""
        self.predictor.save_results()
        
        # Check if files were created
        for model in self.predictor.predictions.keys():
            pred_file = os.path.join(
                config.PREDICTIONS_DIR,
                f"{self.predictor.symbol}_{model}_predictions.csv"
            )
            self.assertTrue(os.path.exists(pred_file))
        
        eval_file = os.path.join(
            config.RESULTS_DIR,
            f"{self.predictor.symbol}_evaluation.csv"
        )
        self.assertTrue(os.path.exists(eval_file))
    
    def test_10_plot_predictions(self):
        """Test visualization functionality"""
        # Temporarily disable showing plots
        show_plots = config.SHOW_PLOTS
        config.SHOW_PLOTS = False
        
        self.predictor.plot_predictions()
        
        # Check if plot file was created
        plot_file = os.path.join(
            config.PLOTS_DIR,
            f"{self.predictor.symbol}_predictions.{config.PLOT_FORMAT}"
        )
        self.assertTrue(os.path.exists(plot_file))
        
        # Restore original setting
        config.SHOW_PLOTS = show_plots

def run_tests():
    """Run all tests"""
    print("🧪 Starting Stock Predictor Tests...")
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == '__main__':
    run_tests()
